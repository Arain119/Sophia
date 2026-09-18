from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic


REPORT_SCHEMA = "sophia_tokenizer_code_corpus_report_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inventory(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list) or not sources:
        raise ValueError("tokenizer code inventory must contain non-empty sources")
    return [dict(source) for source in sources]


def _relative_member_path(name: str) -> str:
    parts = PurePosixPath(name).parts
    if len(parts) < 2:
        return ""
    return PurePosixPath(*parts[1:]).as_posix()


def _selected(path: str, source: dict[str, Any]) -> bool:
    include_prefixes = tuple(str(value) for value in source["include_prefixes"])
    exclude_prefixes = tuple(str(value) for value in source["exclude_prefixes"])
    exclude_suffixes = tuple(str(value) for value in source.get("exclude_suffixes", []))
    extensions = {str(value).lower() for value in source["extensions"]}
    return (
        path.startswith(include_prefixes)
        and not path.startswith(exclude_prefixes)
        and not path.endswith(exclude_suffixes)
        and PurePosixPath(path).suffix.lower() in extensions
    )


def _license_hashes(
    *, archive: tarfile.TarFile, source: dict[str, Any]
) -> dict[str, str]:
    wanted = {str(value) for value in source["license_files"]}
    found: dict[str, str] = {}
    for member in archive.getmembers():
        relative = _relative_member_path(member.name)
        if relative not in wanted or not member.isfile():
            continue
        handle = archive.extractfile(member)
        if handle is not None:
            found[relative] = _sha256_bytes(handle.read())
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"missing license files for {source['name']}: {missing}")
    return dict(sorted(found.items()))


def _source_rows(
    *,
    archive_path: Path,
    source: dict[str, Any],
    seen_hashes: set[str],
    max_file_bytes: int,
    min_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_hash = str(source["archive_sha256"])
    actual_hash = _sha256_file(archive_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"archive hash mismatch for {source['name']}: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    rows: list[dict[str, Any]] = []
    stats = {
        "members_considered": 0,
        "kept_files": 0,
        "kept_utf8_bytes": 0,
        "drop_too_large": 0,
        "drop_binary_or_non_utf8": 0,
        "drop_too_short": 0,
        "drop_exact_duplicate": 0,
        "drop_source_budget": 0,
    }
    max_total_bytes = int(source["max_total_bytes"])
    with tarfile.open(archive_path, mode="r:gz") as archive:
        license_hashes = _license_hashes(archive=archive, source=source)
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            relative = _relative_member_path(member.name)
            if not member.isfile() or not _selected(relative, source):
                continue
            stats["members_considered"] += 1
            if int(member.size) > int(max_file_bytes):
                stats["drop_too_large"] += 1
                continue
            handle = archive.extractfile(member)
            raw = b"" if handle is None else handle.read()
            if b"\x00" in raw:
                stats["drop_binary_or_non_utf8"] += 1
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                stats["drop_binary_or_non_utf8"] += 1
                continue
            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if len(text) < int(min_chars):
                stats["drop_too_short"] += 1
                continue
            content = text.encode("utf-8")
            content_hash = _sha256_bytes(content)
            if content_hash in seen_hashes:
                stats["drop_exact_duplicate"] += 1
                continue
            if stats["kept_utf8_bytes"] + len(content) > max_total_bytes:
                stats["drop_source_budget"] += 1
                continue
            seen_hashes.add(content_hash)
            rows.append(
                {
                    "text": text,
                    "source": str(source["name"]),
                    "repo_id": str(source["repo_id"]),
                    "revision": str(source["revision"]),
                    "license": str(source["license"]),
                    "domain": str(source["domain"]),
                    "path": relative,
                    "content_hash": content_hash,
                }
            )
            stats["kept_files"] += 1
            stats["kept_utf8_bytes"] += len(content)
    return rows, {
        "name": str(source["name"]),
        "repo_id": str(source["repo_id"]),
        "revision": str(source["revision"]),
        "license": str(source["license"]),
        "license_note": str(source["license_note"]),
        "license_file_sha256": license_hashes,
        "archive_file": archive_path.name,
        "archive_sha256": actual_hash,
        "stats": stats,
    }


def build_corpus(
    *,
    inventory_path: Path,
    archives_dir: Path,
    output_dir: Path,
    max_file_bytes: int,
    min_chars: int,
) -> dict[str, Any]:
    sources = _load_inventory(inventory_path)
    seen_hashes: set[str] = set()
    source_reports: list[dict[str, Any]] = []
    for source in sources:
        archive_path = archives_dir / str(source["archive_file"])
        if not archive_path.is_file():
            raise FileNotFoundError(f"missing source archive: {archive_path}")
        rows, source_report = _source_rows(
            archive_path=archive_path,
            source=source,
            seen_hashes=seen_hashes,
            max_file_bytes=max_file_bytes,
            min_chars=min_chars,
        )
        source_output = output_dir / str(source["name"]) / "train"
        source_output.mkdir(parents=True, exist_ok=True)
        output_path = source_output / "part-00000.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows),
            output_path,
            compression="zstd",
            use_dictionary=True,
        )
        source_report["output_path"] = str(output_path.resolve())
        source_report["output_sha256"] = _sha256_file(output_path)
        source_reports.append(source_report)
    report = {
        "schema": REPORT_SCHEMA,
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": _sha256_file(inventory_path),
        "archives_dir": str(archives_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "policy": {
            "split": "train_only_for_tokenizer_training",
            "max_file_bytes": int(max_file_bytes),
            "min_chars": int(min_chars),
            "exact_dedup": "sha256(normalized_utf8_text)",
            "newline_normalization": "crlf_or_cr_to_lf",
        },
        "sources": source_reports,
        "totals": {
            "kept_files": sum(row["stats"]["kept_files"] for row in source_reports),
            "kept_utf8_bytes": sum(
                row["stats"]["kept_utf8_bytes"] for row in source_reports
            ),
        },
    }
    write_json_atomic(str(output_dir / "build_report.json"), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a provenance-rich tokenizer code corpus from pinned archives."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--archives-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-file-bytes", type=int, default=524288)
    parser.add_argument("--min-chars", type=int, default=80)
    args = parser.parse_args(argv)
    report = build_corpus(
        inventory_path=Path(args.inventory),
        archives_dir=Path(args.archives_dir),
        output_dir=Path(args.output_dir),
        max_file_bytes=int(args.max_file_bytes),
        min_chars=int(args.min_chars),
    )
    print(
        f"built tokenizer code corpus files={report['totals']['kept_files']} "
        f"utf8_bytes={report['totals']['kept_utf8_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
