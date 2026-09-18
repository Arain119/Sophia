from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.corpus_quality import dedup_normalize
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file


SELECTION_SCHEMA = "sophia_benchmark_source_selection_v1"
INVENTORY_SCHEMA = "sophia_benchmark_source_inventory_v1"
REPORT_SCHEMA = "sophia_pretrain_decontamination_benchmarks_v2"
CHOICE_KEYS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _json_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=12,
        connect=12,
        read=12,
        status=12,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "Sophia-benchmark-decontamination/1.0"})
    return session


def _selected_pairs(source: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for selection in source.get("selections", []):
        if not isinstance(selection, dict):
            raise ValueError("benchmark selections must be objects")
        config = str(selection.get("config") or "")
        splits = selection.get("splits")
        if not config or not isinstance(splits, list) or not splits:
            raise ValueError("benchmark selection requires config and splits")
        pairs.update((config, str(split)) for split in splits)
    return pairs


def _download_file(
    *,
    session: requests.Session,
    url: str,
    destination: Path,
    expected_bytes: int,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.is_file():
        if int(destination.stat().st_size) != int(expected_bytes):
            raise RuntimeError(f"existing benchmark file size mismatch: {destination}")
    else:
        offset = int(partial.stat().st_size) if partial.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        with session.get(
            url,
            headers=headers,
            stream=True,
            timeout=(15, 120),
        ) as response:
            response.raise_for_status()
            append = offset > 0 and int(response.status_code) == 206
            with partial.open("ab" if append else "wb") as handle:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.replace(destination)
    if int(destination.stat().st_size) != int(expected_bytes):
        raise RuntimeError(f"downloaded benchmark file size mismatch: {destination}")
    return {
        "path": str(destination.resolve()),
        "bytes": int(destination.stat().st_size),
        "sha256": sha256_file(destination),
    }


def snapshot_and_download(
    *,
    selection_path: str,
    raw_root: str,
    inventory_output: str,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    selection_file = Path(selection_path).expanduser().resolve()
    selection = _load_json(selection_file)
    if str(selection.get("schema")) != SELECTION_SCHEMA:
        raise ValueError("unsupported benchmark source selection")
    metadata_endpoint = str(selection.get("metadata_endpoint") or "").rstrip("/")
    parquet_api = str(selection.get("parquet_api") or "")
    if not metadata_endpoint.startswith("https://") or not parquet_api.startswith(
        "https://"
    ):
        raise ValueError("benchmark source endpoints must use https")
    active_session = session or _session()
    root = Path(raw_root).expanduser().resolve()
    sources_out: list[dict[str, Any]] = []
    for source in selection.get("sources", []):
        if not isinstance(source, dict):
            raise ValueError("benchmark source rows must be objects")
        name = str(source.get("name") or "")
        repo_id = str(source.get("repo_id") or "")
        revision = str(source.get("revision") or "")
        license_name = str(source.get("license") or "")
        if not name or not repo_id or len(revision) != 40 or not license_name:
            raise ValueError(f"invalid pinned benchmark source: {name!r}")
        metadata_response = active_session.get(
            f"{metadata_endpoint}/api/datasets/{repo_id}", timeout=(15, 60)
        )
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        if not isinstance(metadata, dict) or str(metadata.get("sha") or "") != revision:
            raise RuntimeError(f"benchmark revision drift: {repo_id}")
        if bool(metadata.get("private")) or bool(metadata.get("disabled")) or metadata.get(
            "gated"
        ) not in (False, None, "false"):
            raise RuntimeError(f"benchmark source is not publicly available: {repo_id}")
        parquet_response = active_session.get(
            parquet_api,
            params={"dataset": repo_id},
            timeout=(15, 60),
        )
        parquet_response.raise_for_status()
        parquet_payload = parquet_response.json()
        if not isinstance(parquet_payload, dict) or parquet_payload.get("failed"):
            raise RuntimeError(f"benchmark parquet conversion failed: {repo_id}")
        selected = _selected_pairs(source)
        available: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in parquet_payload.get("parquet_files", []):
            if not isinstance(item, dict):
                continue
            key = (str(item.get("config") or ""), str(item.get("split") or ""))
            if key in selected:
                available.setdefault(key, []).append(item)
        if set(available) != selected:
            raise RuntimeError(
                f"benchmark parquet selection incomplete: {repo_id} "
                f"missing={sorted(selected - set(available))}"
            )
        files: list[dict[str, Any]] = []
        for config, split in sorted(selected):
            for index, item in enumerate(available[(config, split)]):
                url = str(item.get("url") or "")
                expected_bytes = int(item.get("size") or 0)
                if not url.startswith("https://") or expected_bytes <= 0:
                    raise RuntimeError(f"invalid benchmark parquet file: {repo_id}")
                destination = (
                    root
                    / name
                    / config
                    / split
                    / f"part-{index:05d}.parquet"
                )
                downloaded = _download_file(
                    session=active_session,
                    url=url,
                    destination=destination,
                    expected_bytes=expected_bytes,
                )
                files.append(
                    {
                        "config": config,
                        "split": split,
                        "url": url,
                        **downloaded,
                    }
                )
                print(
                    f"[BENCHMARK] source={name} config={config} split={split} "
                    f"bytes={expected_bytes:,}",
                    flush=True,
                )
        sources_out.append(
            {
                "name": name,
                "repo_id": repo_id,
                "revision": revision,
                "license": license_name,
                "files": files,
            }
        )
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "selection_path": str(selection_file),
        "selection_sha256": _json_sha256(selection_file),
        "raw_root": str(root),
        "sources": sources_out,
        "files": sum(len(source["files"]) for source in sources_out),
        "bytes": sum(
            int(item["bytes"])
            for source in sources_out
            for item in source["files"]
        ),
    }
    write_json_atomic(
        Path(inventory_output).expanduser().resolve(),
        inventory,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return inventory


def _choice_prompt(question: str, labels: Iterable[object], choices: Iterable[object]) -> str:
    lines = [str(question).strip()]
    lines.extend(
        f"{str(label).strip()}. {str(choice).strip()}"
        for label, choice in zip(labels, choices, strict=False)
        if str(choice).strip()
    )
    return "\n".join(lines)


def benchmark_signature(
    *, name: str, row: dict[str, Any]
) -> tuple[str, dict[str, object]] | None:
    prompt = ""
    metadata: dict[str, object] = {}
    if name == "gsm8k":
        prompt = str(row.get("question") or "")
    elif name == "math":
        prompt = str(row.get("problem") or "")
        metadata["subject"] = str(row.get("type") or "")
        metadata["level"] = str(row.get("level") or "")
    elif name == "mmlu":
        choices = row.get("choices") if isinstance(row.get("choices"), list) else []
        prompt = _choice_prompt(
            str(row.get("question") or ""),
            CHOICE_KEYS[: len(choices)],
            choices,
        )
        metadata["subject"] = str(row.get("subject") or "")
    elif name == "arc":
        raw_choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
        labels = raw_choices.get("label") if isinstance(raw_choices.get("label"), list) else []
        texts = raw_choices.get("text") if isinstance(raw_choices.get("text"), list) else []
        prompt = _choice_prompt(str(row.get("question") or ""), labels, texts)
    elif name == "hellaswag":
        endings = row.get("endings") if isinstance(row.get("endings"), list) else []
        prompt = _choice_prompt(
            str(row.get("ctx") or ""),
            CHOICE_KEYS[: len(endings)],
            endings,
        )
        metadata["subject"] = str(row.get("activity_label") or "")
    elif name == "humaneval":
        prompt = str(row.get("prompt") or "")
        metadata["entry_point"] = str(row.get("entry_point") or "")
    else:
        raise ValueError(f"unsupported benchmark normalizer: {name}")
    normalized = dedup_normalize(prompt)
    if len(normalized) < 40:
        return None
    return prompt.strip(), metadata


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def build_signatures(
    *,
    inventory_path: str,
    existing_signatures: list[str],
    output_root: str,
) -> dict[str, Any]:
    inventory_file = Path(inventory_path).expanduser().resolve()
    inventory = _load_json(inventory_file)
    if str(inventory.get("schema")) != INVENTORY_SCHEMA:
        raise ValueError("unsupported benchmark source inventory")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = Counter()
    for existing_value in existing_signatures:
        existing_path = Path(existing_value).expanduser().resolve()
        with existing_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                prompt = str(row.get("prompt") or "")
                normalized = dedup_normalize(prompt)
                if len(normalized) < 40 or normalized in seen:
                    continue
                seen.add(normalized)
                rows.append(dict(row))
                counts[str(row.get("benchmark") or "existing")] += 1
    for source in inventory.get("sources", []):
        if not isinstance(source, dict):
            continue
        name = str(source.get("name") or "")
        for file_info in source.get("files", []):
            if not isinstance(file_info, dict):
                continue
            path = Path(str(file_info.get("path") or "")).resolve()
            if (
                not path.is_file()
                or int(path.stat().st_size) != int(file_info.get("bytes") or 0)
                or sha256_file(path) != str(file_info.get("sha256") or "")
            ):
                raise RuntimeError(f"benchmark inventory file mismatch: {path}")
            table = pq.read_table(path)
            for index, raw in enumerate(table.to_pylist()):
                normalized = benchmark_signature(name=name, row=raw)
                if normalized is None:
                    continue
                prompt, metadata = normalized
                dedup_key = dedup_normalize(prompt)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                rows.append(
                    {
                        "id": (
                            f"{name}:{file_info.get('config')}:{file_info.get('split')}:"
                            f"{index}"
                        ),
                        "benchmark": name,
                        "prompt": prompt,
                        "config": str(file_info.get("config") or ""),
                        "split": str(file_info.get("split") or ""),
                        "revision": str(source.get("revision") or ""),
                        "license": str(source.get("license") or ""),
                        **metadata,
                    }
                )
                counts[name] += 1
    rows.sort(key=lambda row: str(row.get("id") or ""))
    output = Path(output_root).expanduser().resolve()
    signature_path = output / "benchmark_prompt_signatures_v2.jsonl"
    count = _write_jsonl_atomic(signature_path, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "inventory_path": str(inventory_file),
        "inventory_sha256": _json_sha256(inventory_file),
        "existing_signature_files": [
            {
                "path": str(Path(value).expanduser().resolve()),
                "sha256": sha256_file(Path(value).expanduser().resolve()),
            }
            for value in existing_signatures
        ],
        "signature_path": str(signature_path),
        "signature_sha256": sha256_file(signature_path),
        "signatures": count,
        "counts": dict(sorted(counts.items())),
    }
    write_json_atomic(
        output / "decontamination_benchmark_report_v2.json",
        report,
        ensure_ascii=False,
        sort_keys=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire pinned benchmarks and build merged pretrain decontamination signatures."
    )
    parser.add_argument(
        "--selection",
        default="configs/eval/pretrain_decontamination_source_selection.json",
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--inventory-output", required=True)
    parser.add_argument("--existing-signatures", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    snapshot_and_download(
        selection_path=str(args.selection),
        raw_root=str(args.raw_root),
        inventory_output=str(args.inventory_output),
    )
    report = build_signatures(
        inventory_path=str(args.inventory_output),
        existing_signatures=[str(value) for value in args.existing_signatures],
        output_root=str(args.output_root),
    )
    print(
        f"[DONE] signatures={report['signatures']:,} "
        f"seconds={time.time() - started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
