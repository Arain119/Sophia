from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory import INVENTORY_SCHEMA


REPORT_SCHEMA = "sophia_wikimedia_dump_acquisition_v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _file_hashes(path: Path) -> tuple[str, str]:
    sha1 = hashlib.sha1(usedforsecurity=False)
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha1.update(chunk)
            md5.update(chunk)
    return sha1.hexdigest(), md5.hexdigest()


def _verify_file(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    expected_bytes = int(row.get("bytes") or 0)
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Wikimedia file size mismatch: {path} "
            f"expected={expected_bytes} actual={actual_bytes}"
        )
    actual_sha1, actual_md5 = _file_hashes(path)
    if actual_sha1 != str(row.get("sha1") or ""):
        raise ValueError(f"Wikimedia SHA-1 mismatch: {path}")
    if actual_md5 != str(row.get("md5") or ""):
        raise ValueError(f"Wikimedia MD5 mismatch: {path}")
    return {
        "filename": str(row["filename"]),
        "role": str(row["role"]),
        "path": str(path.resolve()),
        "bytes": actual_bytes,
        "sha1": actual_sha1,
        "md5": actual_md5,
        "url": str(row["url"]),
    }


def _download_one(
    *,
    row: dict[str, Any],
    target: Path,
    attempts: int = 20,
) -> dict[str, Any]:
    configured_attempts = str(
        os.environ.get("SOPHIA_WIKIMEDIA_DOWNLOAD_ATTEMPTS") or ""
    ).strip()
    if configured_attempts:
        attempts = int(configured_attempts)
    if not 1 <= int(attempts) <= 10_000:
        raise ValueError("SOPHIA_WIKIMEDIA_DOWNLOAD_ATTEMPTS must be in [1, 10000]")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return _verify_file(target, row)
    partial = target.with_name(f"{target.name}.part")
    expected_bytes = int(row.get("bytes") or 0)
    for attempt in range(1, int(attempts) + 1):
        if partial.is_file() and int(partial.stat().st_size) > expected_bytes:
            raise ValueError(f"Wikimedia partial exceeds expected size: {partial}")
        command = [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--http1.1",
            "--connect-timeout",
            "30",
            "--speed-time",
            "120",
            "--speed-limit",
            "1024",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            str(row["url"]),
        ]
        result = subprocess.run(command, check=False)
        retained = int(partial.stat().st_size) if partial.is_file() else 0
        print(
            f"[WIKIMEDIA-RETRY] attempt={attempt}/{attempts} "
            f"file={target.name} retained_bytes={retained} "
            f"exit={result.returncode}",
            flush=True,
        )
        if result.returncode == 0 and retained == expected_bytes:
            os.replace(partial, target)
            return _verify_file(target, row)
        if retained > expected_bytes:
            raise ValueError(f"Wikimedia download exceeds expected size: {partial}")
        if attempt < int(attempts):
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Wikimedia download attempts exhausted: {target}")


def download_inventory(
    *,
    inventory_path: str,
    output_root: str,
    report_path: str,
    selected_sources: list[str] | None = None,
    downloader: Callable[..., dict[str, Any]] = _download_one,
    workers: int = 1,
) -> dict[str, Any]:
    inventory_file = Path(inventory_path).expanduser().resolve()
    inventory = _load_json(inventory_file)
    if (
        str(inventory.get("schema")) != INVENTORY_SCHEMA
        or str(inventory.get("status")) != "complete"
        or not bool(inventory.get("mutable_latest_urls_forbidden"))
    ):
        raise ValueError("unsupported or incomplete Wikimedia dump inventory")
    requested = {
        str(value).strip() for value in (selected_sources or []) if str(value).strip()
    }
    available = {
        str(source.get("name") or "")
        for source in inventory.get("sources", [])
        if isinstance(source, dict)
    }
    if requested - available:
        raise ValueError(
            f"unknown Wikimedia sources requested: {sorted(requested - available)}"
        )
    worker_count = int(workers)
    if not 1 <= worker_count <= 16:
        raise ValueError("Wikimedia download workers must be between 1 and 16")
    root = Path(output_root).expanduser().resolve()
    selected_sources_rows: list[dict[str, Any]] = []
    work: list[dict[str, Any]] = []
    for source in inventory.get("sources", []):
        if not isinstance(source, dict):
            raise ValueError("Wikimedia inventory source rows must be objects")
        name = str(source.get("name") or "")
        if requested and name not in requested:
            continue
        source_index = len(selected_sources_rows)
        selected_sources_rows.append(source)
        for row in source.get("files", []):
            if not isinstance(row, dict):
                raise ValueError("Wikimedia inventory file rows must be objects")
            target = root / name / str(row["filename"])
            work.append(
                {
                    "source_index": source_index,
                    "source_name": name,
                    "row": row,
                    "target": target,
                }
            )
    if not selected_sources_rows:
        raise ValueError("Wikimedia acquisition selected no sources")

    def _run(item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        print(
            f"[WIKIMEDIA-DOWNLOAD] source={item['source_name']} "
            f"file={item['target'].name}",
            flush=True,
        )
        return int(item["source_index"]), downloader(
            row=item["row"], target=item["target"]
        )

    if worker_count == 1:
        completed = [_run(item) for item in work]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            completed = list(executor.map(_run, work))

    files_by_source: list[list[dict[str, Any]]] = [
        [] for _source in selected_sources_rows
    ]
    for source_index, file_report in completed:
        files_by_source[source_index].append(file_report)

    acquired_sources: list[dict[str, Any]] = []
    for source_index, source in enumerate(selected_sources_rows):
        name = str(source.get("name") or "")
        files = files_by_source[source_index]
        acquired_sources.append(
            {
                "name": name,
                "wiki": str(source.get("wiki") or ""),
                "snapshot_date": str(source.get("snapshot_date") or ""),
                "files": files,
                "acquired_file_count": len(files),
                "acquired_bytes": sum(int(row["bytes"]) for row in files),
            }
        )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "inventory_path": str(inventory_file),
        "inventory_sha256": hashlib.sha256(inventory_file.read_bytes()).hexdigest(),
        "output_root": str(root),
        "download_workers": worker_count,
        "source_count": len(acquired_sources),
        "acquired_file_count": sum(
            int(source["acquired_file_count"]) for source in acquired_sources
        ),
        "acquired_bytes": sum(
            int(source["acquired_bytes"]) for source in acquired_sources
        ),
        "sources": acquired_sources,
    }
    write_json_atomic(
        Path(report_path).expanduser().resolve(),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify an immutable Wikimedia dump inventory."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = download_inventory(
        inventory_path=str(args.inventory),
        output_root=str(args.output_root),
        report_path=str(args.report),
        selected_sources=list(args.source),
        workers=int(args.workers),
    )
    print(
        f"[DONE] sources={report['source_count']} "
        f"files={report['acquired_file_count']} bytes={report['acquired_bytes']:,} "
        f"report={Path(args.report).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
