from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.data.pretrain_document_admission import (
    PostCleanDocumentAdmission,
    load_post_clean_document_admission,
)
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.integrations.adapters.hf.tokenizer import load_local_tokenizer
from ml.tooling.core.pretrain_taxonomy_user import infer_user_taxonomy_label
from ml.tooling.scripts.data.clean_pretrain_corpus import document_time_year
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file


REPORT_SCHEMA = "sophia_pretrain_corpus_token_profile_v1"
STATE_SCHEMA = "sophia_pretrain_corpus_token_profile_state_v1"
TOKEN_LENGTH_THRESHOLDS = (4_096, 8_192, 16_384)
PROFILE_REQUIRED_COLUMNS = {
    "text",
    "source",
    "source_revision",
    "repo_id",
    "license",
    "domain",
    "language",
    "length_bucket",
    "mix_bucket",
    "document_timestamp",
}
_PROFILE_WORKER_TOKENIZER: Any | None = None
_PROFILE_WORKER_ADMISSION: PostCleanDocumentAdmission | None = None


def token_length_bucket(token_count: int) -> str:
    count = int(token_count)
    if count < 4_096:
        return "tokens_lt_4k"
    if count < 8_192:
        return "tokens_4k_8k"
    if count < 16_384:
        return "tokens_8k_16k"
    return "tokens_ge_16k"


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _counter_tree() -> defaultdict[str, Counter[str]]:
    return defaultdict(Counter)


def _normalize_counter_tree(
    value: dict[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    return {
        str(key): {str(inner): int(count) for inner, count in sorted(counter.items())}
        for key, counter in sorted(value.items())
    }


def _merge_counter_tree(
    target: defaultdict[str, Counter[str]],
    source: dict[str, dict[str, int]],
) -> None:
    for dimension, values in source.items():
        target[str(dimension)].update(
            {str(key): int(value) for key, value in values.items()}
        )


def _validate_profile_input(
    path: Path,
    file_info: dict[str, Any],
    *,
    verify_sha256: bool = True,
) -> None:
    if (
        not path.is_file()
        or int(path.stat().st_size) != int(file_info.get("bytes") or 0)
        or (
            verify_sha256
            and sha256_file(path) != str(file_info.get("sha256") or "")
        )
    ):
        raise ValueError(f"clean output fingerprint mismatch: {path}")


def _profile_file(
    path: Path,
    *,
    tokenizer: Any,
    batch_size: int,
    admission: PostCleanDocumentAdmission,
) -> dict[str, Any]:
    file_started = time.time()
    file_documents = Counter()
    file_tokens = Counter()
    file_utf8_bytes = Counter()
    file_exclusions = Counter()
    file_documents_scanned = 0
    file_documents_by = _counter_tree()
    file_tokens_by = _counter_tree()
    file_long_context = {
        str(threshold): {"documents": 0, "tokens": 0}
        for threshold in TOKEN_LENGTH_THRESHOLDS
    }
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    required_columns = PROFILE_REQUIRED_COLUMNS | admission.required_columns
    missing = sorted(required_columns - available)
    if missing:
        raise ValueError(f"profile input is missing columns {missing}: {path}")
    split = path.parent.parent.name
    for batch in parquet.iter_batches(
        batch_size=int(batch_size),
        columns=sorted(required_columns),
    ):
        rows = batch.to_pylist()
        file_documents_scanned += len(rows)
        admitted_rows: list[dict[str, Any]] = []
        for row in rows:
            reason = admission.exclusion_reason(row)
            if reason:
                file_exclusions[reason] += 1
            else:
                admitted_rows.append(row)
        rows = admitted_rows
        if not rows:
            continue
        texts = [str(row.get("text") or "") for row in rows]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        lengths = encoded.get("length")
        if lengths is None:
            lengths = [len(values) for values in encoded["input_ids"]]
        if len(lengths) != len(rows):
            raise RuntimeError("tokenizer length output does not match input batch")
        for row, raw_length, text in zip(rows, lengths, texts, strict=True):
            token_count = int(raw_length) + 1
            byte_count = len(text.encode("utf-8"))
            token_bucket = token_length_bucket(token_count)
            source = str(row.get("source") or "")
            taxonomy = infer_user_taxonomy_label(text)
            time_year = document_time_year(row.get("document_timestamp"))
            dimensions = {
                "split": split,
                "source": source,
                "source_revision": (
                    f"{source}@{str(row.get('source_revision') or '')}"
                ),
                "repo_id": str(row.get("repo_id") or ""),
                "license": str(row.get("license") or ""),
                "language": str(row.get("language") or ""),
                "domain": str(row.get("domain") or ""),
                "character_length_bucket": str(row.get("length_bucket") or ""),
                "token_length_bucket": token_bucket,
                "mix_bucket": str(row.get("mix_bucket") or ""),
                "document_time_year": time_year,
                "source_document_time_year": f"{source}|{time_year}",
                "user_taxonomy": taxonomy,
                "source_user_taxonomy": f"{source}|{taxonomy}",
                "source_split": f"{source}|{split}",
                "mix_bucket_split": (f"{str(row.get('mix_bucket') or '')}|{split}"),
            }
            file_documents["total"] += 1
            file_tokens["total"] += token_count
            file_utf8_bytes["total"] += byte_count
            for dimension, value in dimensions.items():
                file_documents_by[dimension][value] += 1
                file_tokens_by[dimension][value] += token_count
            for threshold in TOKEN_LENGTH_THRESHOLDS:
                if token_count >= threshold:
                    supply = file_long_context[str(threshold)]
                    supply["documents"] += 1
                    supply["tokens"] += token_count
    return {
        "documents_scanned": int(file_documents_scanned),
        "documents_excluded": int(file_exclusions.total()),
        "documents_excluded_by": dict(sorted(file_exclusions.items())),
        "documents": int(file_documents["total"]),
        "tokens": int(file_tokens["total"]),
        "utf8_bytes": int(file_utf8_bytes["total"]),
        "documents_by": _normalize_counter_tree(file_documents_by),
        "tokens_by": _normalize_counter_tree(file_tokens_by),
        "long_context_supply": file_long_context,
        "duration_seconds": float(time.time() - file_started),
    }


def _initialize_profile_worker(
    tokenizer_path: str,
    policy: dict[str, Any],
    policy_path: str,
) -> None:
    global _PROFILE_WORKER_ADMISSION, _PROFILE_WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    _PROFILE_WORKER_TOKENIZER = load_local_tokenizer(
        tokenizer_path,
        model_max_length=1_000_000_000,
    )
    _PROFILE_WORKER_ADMISSION = load_post_clean_document_admission(
        policy,
        policy_path=policy_path,
    )


def _profile_file_worker(
    task: tuple[str, dict[str, Any], int],
) -> dict[str, Any]:
    path_value, file_info, batch_size = task
    if _PROFILE_WORKER_TOKENIZER is None or _PROFILE_WORKER_ADMISSION is None:
        raise RuntimeError("profile worker is not initialized")
    path = Path(path_value)
    _validate_profile_input(path, file_info)
    return _profile_file(
        path,
        tokenizer=_PROFILE_WORKER_TOKENIZER,
        batch_size=int(batch_size),
        admission=_PROFILE_WORKER_ADMISSION,
    )


class _ProfileState:
    def __init__(self, path: Path, *, lineage: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(str(path), timeout=120.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                contribution_json TEXT NOT NULL
            );
            """
        )
        expected = {"state_schema": STATE_SCHEMA, **lineage}
        for key, value in expected.items():
            existing = self.connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (str(key),)
            ).fetchone()
            if existing is not None and str(existing[0]) != str(value):
                self.connection.close()
                raise ValueError(
                    f"token profile resume lineage mismatch for {key}: "
                    f"stored={existing[0]} current={value}"
                )
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                (str(key), str(value)),
            )
        self.connection.commit()

    def get(self, path: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT contribution_json FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[0]))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid token profile resume contribution: {path}")
        return payload

    def put(self, path: str, contribution: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO files(path, contribution_json) VALUES (?, ?)",
            (
                str(path),
                json.dumps(contribution, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def profile_pretrain_corpus(
    *,
    clean_root: str,
    tokenizer_path: str,
    admission_policy: str,
    output_path: str,
    working_state_database: str | None = None,
    batch_size: int = 64,
    workers: int = 1,
    max_files: int = 0,
    verify_cached_file_sha256: bool = True,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    root = Path(clean_root).expanduser().resolve()
    cleaning_report_path = root / "cleaning_report.json"
    cleaning = _load_json(cleaning_report_path)
    if str(cleaning.get("schema")) != "sophia_clean_pretrain_corpus_v1":
        raise ValueError("unsupported clean pretrain corpus report")
    if str(cleaning.get("status")) != "complete":
        raise ValueError("token profiling requires a complete cleaning report")
    policy_path = Path(admission_policy).expanduser().resolve()
    policy = _load_json(policy_path)
    document_admission = load_post_clean_document_admission(
        policy,
        policy_path=policy_path,
    )
    expected_tokenizer_sha1 = str(policy.get("required_tokenizer_bundle_sha1") or "")
    actual_tokenizer_sha1 = compute_tokenizer_bundle_sha1(tokenizer_path)
    if expected_tokenizer_sha1 != actual_tokenizer_sha1:
        raise ValueError(
            "selected tokenizer does not match data admission policy: "
            f"expected={expected_tokenizer_sha1} actual={actual_tokenizer_sha1}"
        )
    tokenizer = (
        load_local_tokenizer(tokenizer_path, model_max_length=1_000_000_000)
        if int(workers) == 1
        else None
    )
    files: list[str] = []
    clean_inventory: dict[str, dict[str, Any]] = {}
    for source in cleaning.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_files = {
            str(_absolute_path(path)) for path in source.get("output_files", [])
        }
        inventory_rows = source.get("output_file_inventory")
        if not isinstance(inventory_rows, list):
            raise ValueError("cleaning report has no output file inventory")
        source_inventory = {
            str(_absolute_path(str(row.get("path") or ""))): row
            for row in inventory_rows
            if isinstance(row, dict)
        }
        if not source_files or set(source_inventory) != source_files:
            raise ValueError(
                "cleaning output file inventory does not match output files"
            )
        overlap = set(clean_inventory) & set(source_inventory)
        if overlap:
            raise ValueError(f"duplicate clean output paths: {sorted(overlap)[:3]}")
        clean_inventory.update(source_inventory)
        files.extend(source_files)
    files = sorted(files)
    if int(max_files) > 0:
        files = files[: int(max_files)]
    if not files:
        raise ValueError("cleaning report contains no output parquet files")
    documents = Counter()
    documents_scanned = Counter()
    documents_excluded_by = Counter()
    tokens = Counter()
    utf8_bytes = Counter()
    documents_by = _counter_tree()
    tokens_by = _counter_tree()
    long_context_supply = {
        str(threshold): {"documents": 0, "tokens": 0}
        for threshold in TOKEN_LENGTH_THRESHOLDS
    }
    cleaning_report_sha256 = hashlib.sha256(
        cleaning_report_path.read_bytes()
    ).hexdigest()
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    selected_files_sha256 = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    taxonomy_path = (
        Path(__file__).resolve().parents[2] / "core" / "pretrain_taxonomy_user.py"
    )
    document_admission_path = (
        Path(__file__).resolve().parents[3] / "data" / "pretrain_document_admission.py"
    )
    output_file = Path(output_path).expanduser().resolve()
    if str(working_state_database or "").strip():
        state_path = Path(str(working_state_database)).expanduser().resolve()
    else:
        state_path = output_file.with_name(output_file.name + ".state.sqlite3")
    state = _ProfileState(
        state_path,
        lineage={
            "cleaning_report_sha256": cleaning_report_sha256,
            "admission_policy_sha256": policy_sha256,
            "tokenizer_bundle_sha1": actual_tokenizer_sha1,
            "selected_files_sha256": selected_files_sha256,
            "profiler_code_sha256": sha256_file(__file__),
            "taxonomy_code_sha256": sha256_file(taxonomy_path),
            "document_admission_code_sha256": sha256_file(document_admission_path),
        },
    )
    cumulative_duration = 0.0
    executor: ProcessPoolExecutor | None = None

    def merge_contribution(file_index: int, contribution: dict[str, Any]) -> None:
        nonlocal cumulative_duration
        documents_scanned["total"] += int(
            contribution.get("documents_scanned", contribution["documents"])
        )
        documents_excluded_by.update(
            {
                str(reason): int(count)
                for reason, count in dict(
                    contribution.get("documents_excluded_by") or {}
                ).items()
            }
        )
        documents["total"] += int(contribution["documents"])
        tokens["total"] += int(contribution["tokens"])
        utf8_bytes["total"] += int(contribution["utf8_bytes"])
        _merge_counter_tree(documents_by, contribution["documents_by"])
        _merge_counter_tree(tokens_by, contribution["tokens_by"])
        for threshold in TOKEN_LENGTH_THRESHOLDS:
            key = str(threshold)
            long_context_supply[key]["documents"] += int(
                contribution["long_context_supply"][key]["documents"]
            )
            long_context_supply[key]["tokens"] += int(
                contribution["long_context_supply"][key]["tokens"]
            )
        cumulative_duration += float(contribution["duration_seconds"])
        if file_index % 25 == 0 or file_index == len(files):
            print(
                f"[PROFILE] files={file_index}/{len(files)} "
                f"documents={documents['total']:,} tokens={tokens['total']:,}",
                flush=True,
            )

    try:
        if int(workers) > 1:
            executor = ProcessPoolExecutor(
                max_workers=int(workers),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_profile_worker,
                initargs=(
                    str(Path(tokenizer_path).expanduser().resolve()),
                    policy,
                    str(policy_path),
                ),
            )
        chunk_size = max(1, 2 * int(workers))
        for chunk_start in range(0, len(files), chunk_size):
            pending: list[
                tuple[
                    int,
                    Path,
                    dict[str, Any] | None,
                    Future[dict[str, Any]] | None,
                ]
            ] = []
            for offset, file_value in enumerate(
                files[chunk_start : chunk_start + chunk_size]
            ):
                file_index = chunk_start + offset + 1
                path = Path(file_value)
                path_key = str(_absolute_path(path))
                file_info = clean_inventory[path_key]
                contribution = state.get(path_key)
                future: Future[dict[str, Any]] | None = None
                if contribution is not None:
                    _validate_profile_input(
                        path,
                        file_info,
                        verify_sha256=bool(verify_cached_file_sha256),
                    )
                elif executor is None:
                    if tokenizer is None:
                        raise RuntimeError("profile tokenizer is not initialized")
                    _validate_profile_input(path, file_info)
                    contribution = _profile_file(
                        path,
                        tokenizer=tokenizer,
                        batch_size=int(batch_size),
                        admission=document_admission,
                    )
                    state.put(path_key, contribution)
                else:
                    future = executor.submit(
                        _profile_file_worker,
                        (path_key, file_info, int(batch_size)),
                    )
                pending.append((file_index, path, contribution, future))
            for file_index, path, contribution, future in pending:
                if contribution is None:
                    if future is None:
                        raise RuntimeError("profile contribution has no worker future")
                    contribution = future.result()
                    state.put(str(_absolute_path(path)), contribution)
                merge_contribution(file_index, contribution)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        state.close()
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "clean_root": str(root),
        "cleaning_report": str(cleaning_report_path),
        "cleaning_report_sha256": cleaning_report_sha256,
        "admission_policy": str(policy_path),
        "admission_policy_sha256": policy_sha256,
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "tokenizer_bundle_sha1": actual_tokenizer_sha1,
        "eos_tokens_per_document": 1,
        "batch_size": int(batch_size),
        "workers": int(workers),
        "cached_file_validation": (
            "size_and_sha256"
            if verify_cached_file_sha256
            else "size_against_previously_sha256_verified_clean_inventory"
        ),
        "profiled_files": len(files),
        "documents_scanned": int(documents_scanned["total"]),
        "documents_excluded": int(documents_excluded_by.total()),
        "documents_excluded_by": dict(sorted(documents_excluded_by.items())),
        "post_clean_document_admission": document_admission.to_report(),
        "documents": int(documents["total"]),
        "tokens": int(tokens["total"]),
        "utf8_bytes": int(utf8_bytes["total"]),
        "duration_seconds": cumulative_duration,
        "resume_state": str(state_path),
        "documents_by": _normalize_counter_tree(documents_by),
        "tokens_by": _normalize_counter_tree(tokens_by),
        "long_context_supply": long_context_supply,
    }
    write_json_atomic(
        output_file,
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile exact token supply in the cleaned pretrain corpus."
    )
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--tokenizer", default="ml/modeling/text")
    parser.add_argument(
        "--admission-policy",
        default="configs/data/pretrain_data_admission_policy.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--working-state-database",
        default=None,
        help=(
            "Optional SQLite resume-state path. Production runs should place this "
            "on a Linux-native filesystem even when the final report is on bulk storage."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--trust-cached-file-sha256",
        action="store_true",
        help=(
            "For cached contributions, validate file existence and size without "
            "rehashing bytes already covered by a verified clean inventory."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = profile_pretrain_corpus(
        clean_root=str(args.clean_root),
        tokenizer_path=str(args.tokenizer),
        admission_policy=str(args.admission_policy),
        output_path=str(args.output),
        working_state_database=(
            None
            if args.working_state_database is None
            else str(args.working_state_database)
        ),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        max_files=int(args.max_files),
        verify_cached_file_sha256=not bool(args.trust_cached_file_sha256),
    )
    print(
        f"[DONE] documents={report['documents']:,} tokens={report['tokens']:,} "
        f"output={Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
