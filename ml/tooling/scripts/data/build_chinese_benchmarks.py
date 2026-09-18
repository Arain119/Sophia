from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.download_hf_source_inventory import (
    _download_https,
    sha256_file,
)


CEVAL_REPO = "ceval/ceval-exam"
CEVAL_REVISION = "617524a00b307ff6f9933702f724131fe12ca7ce"
CEVAL_PARQUET_REVISION = "8267189d6ba0d516d414a98919558032958c4466"
CEVAL_LICENSE = "cc-by-nc-sa-4.0"
CMMLU_REPO = "lmlmcat/cmmlu"
CMMLU_REVISION = "efcc940752ea4a1ea94d2727f11f83858d64fc8e"
CMMLU_LICENSE = "cc-by-nc-4.0"
CMMLU_ARCHIVE = "cmmlu_v1_0_1.zip"
CHOICE_KEYS = ("A", "B", "C", "D")


def ceval_file_inventory(attempts: int = 12) -> list[str]:
    url = "https://datasets-server.huggingface.co/parquet?dataset=ceval%2Fceval-exam"
    for attempt in range(1, int(attempts) + 1):
        try:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            payload = response.json()
            paths = {
                f"{item['config']}/{item['split']}/0000.parquet"
                for item in payload["parquet_files"]
            }
            if len(paths) != 156:
                raise ValueError(f"expected 156 C-Eval parquet files, found {len(paths)}")
            return sorted(paths)
        except (KeyError, ValueError, requests.RequestException) as exc:
            if attempt >= int(attempts):
                raise
            print(
                f"[RETRY] C-Eval inventory attempt={attempt}/{attempts} "
                f"error={type(exc).__name__}",
                flush=True,
            )
            time.sleep(min(2**attempt, 15))
    raise RuntimeError("unreachable C-Eval inventory state")


def format_prompt(question: str, choices: dict[str, str]) -> str:
    lines = [str(question).strip()]
    lines.extend(
        f"{key}. {str(choices.get(key) or '').strip()}"
        for key in CHOICE_KEYS
        if str(choices.get(key) or "").strip()
    )
    return "\n".join(lines)


def normalize_row(
    *,
    benchmark: str,
    subject: str,
    split: str,
    row_id: object,
    row: dict[str, Any],
    revision: str,
    license_name: str,
) -> dict[str, Any] | None:
    question = str(row.get("question") or row.get("Question") or "").strip()
    if not question:
        return None
    choices = {
        key: str(row.get(key) or row.get(key.lower()) or "").strip()
        for key in CHOICE_KEYS
    }
    answer = str(row.get("answer") or row.get("Answer") or "").strip().upper()
    if answer not in CHOICE_KEYS:
        answer = ""
    return {
        "id": f"{benchmark}:{subject}:{split}:{row_id}",
        "benchmark": benchmark,
        "subject": subject,
        "split": split,
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": format_prompt(question, choices),
        "revision": revision,
        "license": license_name,
    }


def read_ceval(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(root.rglob("*.parquet")):
        if path.parent.name in {"dev", "test", "val"}:
            subject = path.parent.parent.name
            split = path.parent.name
        else:
            subject = path.parent.name
            split = path.name.split("-", 1)[0]
        for index, raw in enumerate(pq.read_table(path).to_pylist()):
            item = normalize_row(
                benchmark="ceval",
                subject=subject,
                split=split,
                row_id=raw.get("id", index),
                row=raw,
                revision=CEVAL_REVISION,
                license_name=CEVAL_LICENSE,
            )
            if item is not None:
                yield item


def _cmmlu_split(path: Path) -> str:
    lowered = {part.lower() for part in path.parts}
    if "dev" in lowered:
        return "dev"
    if "test" in lowered:
        return "test"
    return path.stem.rsplit("_", 1)[-1].lower()


def read_cmmlu(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(root.rglob("*.csv")):
        split = _cmmlu_split(path)
        subject = path.stem
        for suffix in ("_dev", "_test"):
            if subject.endswith(suffix):
                subject = subject[: -len(suffix)]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, raw in enumerate(csv.DictReader(handle)):
                item = normalize_row(
                    benchmark="cmmlu",
                    subject=subject,
                    split=split,
                    row_id=raw.get("id", index),
                    row=raw,
                    revision=CMMLU_REVISION,
                    license_name=CMMLU_LICENSE,
                )
                if item is not None:
                    yield item


def download_sources(raw_root: Path) -> dict[str, Any]:
    ceval_root = raw_root / "ceval_parquet"
    cmmlu_root = raw_root / "cmmlu"
    ceval_files = ceval_file_inventory()
    def download_ceval(filename: str) -> str:
        return _download_https(
            repo_id=CEVAL_REPO,
            revision=CEVAL_PARQUET_REVISION,
            filename=filename,
            local_dir=str(ceval_root),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_ceval, filename): filename
            for filename in ceval_files
        }
        for index, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            future.result()
            print(
                f"[BENCHMARK] ceval completed={index}/{len(ceval_files)} {filename}",
                flush=True,
            )
    archive = Path(
        _download_https(
            repo_id=CMMLU_REPO,
            revision=CMMLU_REVISION,
            filename=CMMLU_ARCHIVE,
            local_dir=str(cmmlu_root),
        )
    )
    extract_root = cmmlu_root / "extracted"
    if not extract_root.is_dir() or not any(extract_root.rglob("*.csv")):
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract_root)
    return {
        "ceval_root": str(ceval_root),
        "ceval_files": len(ceval_files),
        "cmmlu_root": str(extract_root),
        "cmmlu_archive_sha256": sha256_file(archive),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_benchmarks(*, raw_root: str, output_root: str) -> dict[str, Any]:
    raw = Path(raw_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    acquisition = download_sources(raw)
    rows = list(read_ceval(Path(acquisition["ceval_root"])))
    rows.extend(read_cmmlu(Path(acquisition["cmmlu_root"])))
    rows.sort(key=lambda item: str(item["id"]))
    benchmark_path = output / "chinese_multiple_choice.jsonl"
    write_jsonl(benchmark_path, rows)
    signature_rows = []
    for row in rows:
        question = str(row["question"])
        signature = (
            question
            if len("".join(question.split())) >= 40
            else str(row["prompt"])
        )
        if len("".join(signature.split())) >= 40:
            signature_rows.append(
                {
                    "id": str(row["id"]),
                    "prompt": signature,
                    "benchmark": str(row["benchmark"]),
                }
            )
    signatures_path = output / "benchmark_prompt_signatures.jsonl"
    signature_count = write_jsonl(signatures_path, signature_rows)
    counts: dict[str, int] = {}
    scored_counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['benchmark']}:{row['split']}"
        counts[key] = counts.get(key, 0) + 1
        if row["answer"]:
            scored_counts[key] = scored_counts.get(key, 0) + 1
    report = {
        "schema": "sophia_chinese_benchmarks_v1",
        "sources": {
            "ceval": {
                "repo_id": CEVAL_REPO,
                "revision": CEVAL_REVISION,
                "parquet_revision": CEVAL_PARQUET_REVISION,
                "license": CEVAL_LICENSE,
            },
            "cmmlu": {
                "repo_id": CMMLU_REPO,
                "revision": CMMLU_REVISION,
                "license": CMMLU_LICENSE,
            },
        },
        "acquisition": acquisition,
        "benchmark_jsonl": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "signature_jsonl": str(signatures_path),
        "signature_sha256": sha256_file(signatures_path),
        "rows": len(rows),
        "signatures": signature_count,
        "counts": dict(sorted(counts.items())),
        "scored_counts": dict(sorted(scored_counts.items())),
    }
    write_json_atomic(
        output / "benchmark_report.json",
        report,
        ensure_ascii=False,
        sort_keys=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pinned C-Eval and CMMLU assets")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_benchmarks(
        raw_root=str(args.raw_root),
        output_root=str(args.output_root),
    )
    print(
        f"[DONE] benchmark_rows={report['rows']:,} signatures={report['signatures']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
