from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any
import uuid

import numpy as np
import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.data.pretrain_document_admission import (
    PostCleanDocumentAdmission,
    load_post_clean_document_admission,
)
from ml.data.token_shards.shard_manifest import (
    TokenShard,
    TokenShardManifest,
    save_manifest,
    sha1_file,
)
from ml.data.token_shards.tokenizer_fingerprint import (
    compute_tokenizer_bundle_sha1,
)
from ml.tooling.scripts.data.clean_pretrain_corpus import (
    document_split,
    document_time_year,
)
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file
from ml.training.pretrain.data_admission import validate_pretrain_data_admission
from ml.training.pretrain.shard_builder_encode import _tokenize_texts
from ml.training.pretrain.shard_builder_output import (
    _promote_staging_out_dir,
    _write_output_dir_marker,
)
from ml.training.pretrain.shard_builder_tokenizer import _load_tokenizer
from ml.training.pretrain.shard_builder_writer import ShardWriter


DATASET_SCHEMA = "sophia_pretrain_token_dataset_v1"
SPLIT_REPORT_SCHEMA = "sophia_pretrain_token_split_v1"
SPLIT_BUILD_STATE_SCHEMA = "sophia_pretrain_token_split_build_state_v1"
ROOT_MARKER = ".sophia_pretrain_token_dataset.json"
REQUIRED_DIMENSIONS = (
    "source",
    "source_revision",
    "repo_id",
    "license",
    "language",
    "domain",
    "character_length_bucket",
    "token_length_bucket",
    "mix_bucket",
    "document_time_year",
    "source_document_time_year",
)
TOKEN_LENGTH_THRESHOLDS = (4_096, 8_192, 16_384)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _token_length_bucket(token_count: int) -> str:
    count = int(token_count)
    if count < 4_096:
        return "tokens_lt_4k"
    if count < 8_192:
        return "tokens_4k_8k"
    if count < 16_384:
        return "tokens_8k_16k"
    return "tokens_ge_16k"


def _counter_tree() -> defaultdict[str, Counter[str]]:
    return defaultdict(Counter)


def _normalize_counter_tree(
    counters: dict[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    return {
        name: {key: int(value) for key, value in sorted(counter.items())}
        for name, counter in sorted(counters.items())
    }


def _restore_counter_tree(
    value: object,
) -> defaultdict[str, Counter[str]]:
    restored = _counter_tree()
    if not isinstance(value, dict):
        raise ValueError("split build state counter tree must be an object")
    for dimension, raw_counts in value.items():
        if not isinstance(raw_counts, dict):
            raise ValueError("split build state counter values must be objects")
        restored[str(dimension)].update(
            {str(key): int(count) for key, count in raw_counts.items()}
        )
    return restored


def _stable_source_seed(seed: int, split: str, source: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{split}\0{source}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _split_from_clean_path(path: Path) -> str:
    if len(path.parents) < 3:
        return ""
    return str(path.parent.parent.name)


def _files_by_source_split(
    *, clean_root: Path, cleaning: dict[str, Any]
) -> dict[str, dict[str, list[Path]]]:
    result: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for source in cleaning.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_name = str(source.get("name") or "")
        if not source_name:
            raise ValueError("cleaning report contains an unnamed source")
        output_files = {
            str(Path(raw_path).resolve()) for raw_path in source.get("output_files", [])
        }
        inventory_rows = source.get("output_file_inventory")
        if not isinstance(inventory_rows, list):
            raise ValueError("cleaning report has no output file inventory")
        inventory = {
            str(Path(str(row.get("path") or "")).resolve()): row
            for row in inventory_rows
            if isinstance(row, dict)
        }
        if not output_files or set(inventory) != output_files:
            raise ValueError(
                "cleaning output file inventory does not match output files"
            )
        for raw_path in sorted(output_files):
            path = Path(str(raw_path)).expanduser().resolve()
            if path != clean_root and clean_root not in path.parents:
                raise ValueError(f"clean output escapes clean_root: {path}")
            split = _split_from_clean_path(path)
            if split not in {"train", "val", "test"}:
                raise ValueError(f"unable to infer clean split from path: {path}")
            file_info = inventory[str(path)]
            if (
                not path.is_file()
                or int(path.stat().st_size) != int(file_info.get("bytes") or 0)
                or sha256_file(path) != str(file_info.get("sha256") or "")
            ):
                raise ValueError(f"clean output fingerprint mismatch: {path}")
            result[source_name][split].append(path)
    return {
        source: {split: sorted(paths) for split, paths in sorted(split_map.items())}
        for source, split_map in sorted(result.items())
    }


def _validate_inputs(
    *,
    clean_root: str,
    cleaning_report_path: str = "",
    profile_path: str,
    mix_plan_path: str,
    tokenizer_path: str,
    admission_policy: str,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, str],
]:
    root = Path(clean_root).expanduser().resolve()
    validate_pretrain_data_admission(
        data_path=str(root),
        policy_path=admission_policy,
    )
    canonical_cleaning_path = root / "cleaning_report.json"
    cleaning_path = (
        Path(str(cleaning_report_path or canonical_cleaning_path))
        .expanduser()
        .resolve()
    )
    cleaning = _load_json(cleaning_path)
    if (
        str(cleaning.get("schema")) != "sophia_clean_pretrain_corpus_v1"
        or str(cleaning.get("status")) != "complete"
    ):
        raise ValueError("shard build requires a complete clean corpus report")
    if Path(str(cleaning.get("output_root") or "")).resolve() != root:
        raise ValueError("cleaning report output_root does not match clean_root")

    profile_file = Path(profile_path).expanduser().resolve()
    profile = _load_json(profile_file)
    if (
        str(profile.get("schema")) != "sophia_pretrain_corpus_token_profile_v1"
        or str(profile.get("status")) != "complete"
    ):
        raise ValueError("shard build requires a complete token profile")
    if Path(str(profile.get("clean_root") or "")).resolve() != root:
        raise ValueError("token profile clean_root does not match shard input")
    profile_cleaning_path = (
        Path(str(profile.get("cleaning_report") or "")).expanduser().resolve()
    )
    if profile_cleaning_path != cleaning_path:
        if not str(cleaning_report_path).strip() or (
            profile_cleaning_path != canonical_cleaning_path
        ):
            raise ValueError(
                "token profile does not reference the selected cleaning report"
            )

    mix_file = Path(mix_plan_path).expanduser().resolve()
    mix = _load_json(mix_file)
    if (
        str(mix.get("schema")) != "sophia_resolved_pretrain_mix_plan_v1"
        or str(mix.get("status")) != "ready"
    ):
        raise ValueError("shard build requires a ready resolved mix plan")
    requested = int(mix.get("requested_unique_train_tokens") or 0)
    resolved = int(mix.get("resolved_unique_train_tokens") or 0)
    if requested <= 0 or resolved != requested:
        raise ValueError(
            "resolved mix does not satisfy the requested unique token budget"
        )

    policy_file = Path(admission_policy).expanduser().resolve()
    policy = _load_json(policy_file)
    cleaning_sha256 = _sha256_path(cleaning_path)
    profile_sha256 = _sha256_path(profile_file)
    policy_sha256 = _sha256_path(policy_file)
    if str(profile.get("cleaning_report_sha256") or "") != cleaning_sha256:
        raise ValueError("token profile cleaning report fingerprint mismatch")
    if str(profile.get("admission_policy_sha256") or "") != policy_sha256:
        raise ValueError("token profile admission policy fingerprint mismatch")
    if str(mix.get("profile_sha256") or "") != profile_sha256:
        raise ValueError("resolved mix token profile fingerprint mismatch")
    mix_policy_value = str(mix.get("mix_policy_path") or "").strip()
    if not mix_policy_value:
        raise ValueError("resolved mix has no pinned mix policy")
    mix_policy_file = Path(mix_policy_value).expanduser().resolve()
    if not mix_policy_file.is_file() or str(mix.get("mix_policy_sha256") or "") != (
        _sha256_path(mix_policy_file)
    ):
        raise ValueError("resolved mix policy fingerprint mismatch")
    expected_tokenizer = str(policy.get("required_tokenizer_bundle_sha1") or "")
    tokenizer_sha1 = compute_tokenizer_bundle_sha1(tokenizer_path)
    fingerprints = {
        "cleaning_report_sha256": cleaning_sha256,
        "profile_sha256": profile_sha256,
        "mix_plan_sha256": _sha256_path(mix_file),
        "mix_policy_sha256": _sha256_path(mix_policy_file),
        "admission_policy_sha256": policy_sha256,
    }
    recorded_tokenizers = {
        expected_tokenizer,
        str(profile.get("tokenizer_bundle_sha1") or ""),
        str(mix.get("tokenizer_bundle_sha1") or ""),
    }
    if recorded_tokenizers != {tokenizer_sha1}:
        raise ValueError(
            "tokenizer lineage mismatch across admission policy, profile, and mix plan"
        )

    source_quotas = {
        str(key): int(value)
        for key, value in dict(mix.get("source_train_quotas") or {}).items()
    }
    composite_quotas = {
        str(key): int(value)
        for key, value in dict(
            mix.get("composite_source_character_length_quotas") or {}
        ).items()
    }
    if (
        sum(source_quotas.values()) != requested
        or sum(composite_quotas.values()) != requested
    ):
        raise ValueError("resolved mix quotas do not sum to the requested token budget")
    composite_by_source = Counter()
    for key, value in composite_quotas.items():
        source, separator, _bucket = key.partition("|")
        if not separator or value <= 0:
            raise ValueError(f"invalid composite mix quota: {key}={value}")
        composite_by_source[source] += value
    if dict(composite_by_source) != source_quotas:
        raise ValueError("composite mix quotas do not reproduce source quotas")
    return root, cleaning, profile, mix, tokenizer_sha1, fingerprints


def _split_quotas(
    *, split: str, profile: dict[str, Any], mix: dict[str, Any]
) -> dict[str, int]:
    if split == "train":
        raw = mix.get("composite_source_character_length_quotas") or {}
        return {str(key): int(value) for key, value in dict(raw).items()}
    supply = profile.get("tokens_by", {}).get("mix_bucket_split", {})
    if not isinstance(supply, dict):
        raise ValueError("token profile is missing mix_bucket_split supply")
    suffix = f"|{split}"
    quotas = {
        str(key)[: -len(suffix)]: int(value)
        for key, value in supply.items()
        if str(key).endswith(suffix) and int(value) > 0
    }
    if not quotas:
        raise ValueError(f"token profile has no {split} supply")
    return quotas


def _accepted_tokens(
    ids: list[int], *, eos_id: int, remaining: int
) -> tuple[np.ndarray, bool]:
    full_count = len(ids) + 1
    if full_count <= int(remaining):
        return np.asarray([*ids, int(eos_id)], dtype=np.int32), False
    if int(remaining) <= 0:
        return np.empty((0,), dtype=np.int32), False
    if int(remaining) == 1:
        return np.asarray([int(eos_id)], dtype=np.int32), True
    return np.asarray(
        [*ids[: int(remaining) - 1], int(eos_id)],
        dtype=np.int32,
    ), True


def _row_dimensions(row: dict[str, Any], *, token_count: int) -> dict[str, str]:
    source = str(row.get("source") or "")
    time_year = document_time_year(row.get("document_timestamp"))
    return {
        "source": source,
        "source_revision": f"{source}@{str(row.get('source_revision') or '')}",
        "repo_id": str(row.get("repo_id") or ""),
        "license": str(row.get("license") or ""),
        "language": str(row.get("language") or ""),
        "domain": str(row.get("domain") or ""),
        "character_length_bucket": str(row.get("length_bucket") or ""),
        "token_length_bucket": _token_length_bucket(token_count),
        "mix_bucket": str(row.get("mix_bucket") or ""),
        "document_time_year": time_year,
        "source_document_time_year": f"{source}|{time_year}",
    }


def _admitted_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    split: str,
    path: Path,
    document_admission: PostCleanDocumentAdmission,
    exclusions: Counter[str],
) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("source") or "") != source:
            raise ValueError(f"row source mismatch in {path}")
        document_hash = str(row.get("document_sha256") or "")
        if document_split(document_hash) != split:
            raise ValueError(f"document split hash mismatch in {path}")
        reason = document_admission.exclusion_reason(row)
        if reason:
            exclusions[reason] += 1
        else:
            admitted.append(row)
    return admitted


def _split_staging_dir(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}.building")


def _split_state_identity(
    *,
    split: str,
    quotas: dict[str, int],
    tokenizer_sha1: str,
    shard_size_tokens: int,
    batch_size: int,
    checkpoint_every_files: int,
    seed: int,
    fingerprints: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": SPLIT_BUILD_STATE_SCHEMA,
        "split": str(split),
        "seed": int(seed),
        "tokenizer_bundle_sha1": str(tokenizer_sha1),
        "shard_size_tokens": int(shard_size_tokens),
        "batch_size": int(batch_size),
        "checkpoint_every_files": int(checkpoint_every_files),
        "target_tokens_by_mix_bucket": dict(sorted(quotas.items())),
        "lineage": dict(fingerprints),
    }


def _validate_committed_shards(
    *,
    staging: Path,
    raw_inventory: object,
) -> tuple[list[TokenShard], list[dict[str, object]]]:
    if not isinstance(raw_inventory, list):
        raise ValueError("split build state shard inventory must be a list")
    shards: list[TokenShard] = []
    inventory: list[dict[str, object]] = []
    committed_paths: set[str] = set()
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, dict):
            raise ValueError("split build state shard entries must be objects")
        expected_name = f"shard_{index:05d}.bin"
        name = str(raw.get("path") or "")
        tokens = int(raw.get("tokens") or 0)
        byte_count = int(raw.get("bytes") or 0)
        expected_sha256 = str(raw.get("sha256") or "")
        path = staging / name
        if (
            name != expected_name
            or tokens <= 0
            or byte_count != tokens * np.dtype("int32").itemsize
            or not path.is_file()
            or int(path.stat().st_size) != byte_count
            or sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"split build state shard fingerprint mismatch: {path}")
        shards.append(TokenShard(path=name, tokens=tokens))
        inventory.append(
            {
                "path": name,
                "tokens": tokens,
                "bytes": byte_count,
                "sha256": expected_sha256,
            }
        )
        committed_paths.add(name)
    for path in staging.glob("shard_*.bin"):
        if path.name not in committed_paths:
            path.unlink()
    for path in staging.glob("shard_*.bin.tmp.*"):
        path.unlink()
    return shards, inventory


def _new_split_build_state(
    *,
    identity: dict[str, Any],
    quotas: dict[str, int],
) -> dict[str, Any]:
    return {
        **identity,
        "status": "building",
        "remaining_tokens_by_mix_bucket": dict(sorted(quotas.items())),
        "processed_files": [],
        "documents_read": 0,
        "documents_excluded": 0,
        "documents_excluded_by": {},
        "documents_selected": 0,
        "quota_boundary_prefix_truncations": 0,
        "tokens_by": {},
        "documents_by": {},
        "shards": [],
        "elapsed_seconds": 0.0,
    }


def _load_or_create_split_build_state(
    *,
    staging: Path,
    identity: dict[str, Any],
    quotas: dict[str, int],
) -> tuple[dict[str, Any], list[TokenShard], list[dict[str, object]]]:
    state_path = staging / "build_state.json"
    if state_path.is_file():
        state = _load_json(state_path)
        for key, expected in identity.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"split build resume identity mismatch for {key}: "
                    f"stored={state.get(key)!r} current={expected!r}"
                )
        remaining = state.get("remaining_tokens_by_mix_bucket")
        if not isinstance(remaining, dict) or set(remaining) != set(quotas):
            raise ValueError("split build state quota keys do not match current quotas")
        if any(int(value) < 0 for value in remaining.values()):
            raise ValueError("split build state has negative remaining quota")
        processed = state.get("processed_files")
        if not isinstance(processed, list) or len(processed) != len(set(processed)):
            raise ValueError("split build state has invalid processed_files")
        shards, inventory = _validate_committed_shards(
            staging=staging,
            raw_inventory=state.get("shards"),
        )
        return state, shards, inventory
    if any(staging.iterdir()):
        raise ValueError(f"split staging directory has no resumable state: {staging}")
    state = _new_split_build_state(identity=identity, quotas=quotas)
    write_json_atomic(state_path, state, ensure_ascii=False, sort_keys=True)
    return state, [], []


def _updated_shard_inventory(
    *,
    staging: Path,
    shards: list[TokenShard],
    committed: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(shards) < len(committed):
        raise RuntimeError("shard writer lost committed shards")
    inventory = list(committed)
    for index, shard in enumerate(shards[len(committed) :], start=len(committed)):
        expected_name = f"shard_{index:05d}.bin"
        if shard.path != expected_name:
            raise RuntimeError("shard writer sequence drifted during resumable build")
        path = staging / shard.path
        byte_count = int(path.stat().st_size)
        if byte_count != int(shard.tokens) * np.dtype("int32").itemsize:
            raise RuntimeError(f"shard byte size does not match token count: {path}")
        inventory.append(
            {
                "path": shard.path,
                "tokens": int(shard.tokens),
                "bytes": byte_count,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _write_split(
    *,
    split: str,
    files_by_source: dict[str, dict[str, list[Path]]],
    quotas: dict[str, int],
    tokenizer: Any,
    tokenizer_sha1: str,
    output_dir: Path,
    shard_size_tokens: int,
    batch_size: int,
    checkpoint_every_files: int,
    seed: int,
    fingerprints: dict[str, str],
    document_admission: PostCleanDocumentAdmission,
) -> dict[str, Any]:
    if int(checkpoint_every_files) <= 0:
        raise ValueError("checkpoint_every_files must be > 0")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"split output is non-empty without a report: {output_dir}")
    staging = _split_staging_dir(output_dir)
    staging.mkdir(parents=True, exist_ok=True)
    identity = _split_state_identity(
        split=split,
        quotas=quotas,
        tokenizer_sha1=tokenizer_sha1,
        shard_size_tokens=int(shard_size_tokens),
        batch_size=int(batch_size),
        checkpoint_every_files=int(checkpoint_every_files),
        seed=int(seed),
        fingerprints=fingerprints,
    )
    state, committed_shards, committed_inventory = _load_or_create_split_build_state(
        staging=staging,
        identity=identity,
        quotas=quotas,
    )
    writer = ShardWriter(
        str(staging),
        shard_size_tokens=int(shard_size_tokens),
        dtype=np.dtype("int32"),
    )
    writer.seed_resume(
        shard_idx=len(committed_shards),
        shards=committed_shards,
    )
    remaining = {
        str(key): int(value)
        for key, value in dict(
            state.get("remaining_tokens_by_mix_bucket") or {}
        ).items()
    }
    tokens_by = _restore_counter_tree(state.get("tokens_by") or {})
    documents_by = _restore_counter_tree(state.get("documents_by") or {})
    documents_read = int(state.get("documents_read") or 0)
    documents_excluded_by = Counter(
        {
            str(reason): int(count)
            for reason, count in dict(state.get("documents_excluded_by") or {}).items()
        }
    )
    documents_selected = int(state.get("documents_selected") or 0)
    prefix_truncations = int(state.get("quota_boundary_prefix_truncations") or 0)
    processed_files = [str(value) for value in state.get("processed_files") or []]
    processed_set = set(processed_files)
    eligible_files = {
        str(path.resolve())
        for source in {key.partition("|")[0] for key in quotas}
        for path in files_by_source.get(source, {}).get(split, [])
    }
    unknown_processed = sorted(processed_set - eligible_files)
    if unknown_processed:
        raise ValueError(
            "split build state references files outside the current input inventory: "
            f"{unknown_processed[:3]}"
        )
    elapsed_before = float(state.get("elapsed_seconds") or 0.0)
    files_since_checkpoint = 0
    started = time.time()

    def checkpoint(*, status: str = "building") -> None:
        nonlocal committed_inventory, files_since_checkpoint
        writer.flush_shard()
        committed_inventory = _updated_shard_inventory(
            staging=staging,
            shards=writer.shards,
            committed=committed_inventory,
        )
        payload = {
            **identity,
            "status": str(status),
            "remaining_tokens_by_mix_bucket": dict(sorted(remaining.items())),
            "processed_files": list(processed_files),
            "documents_read": int(documents_read),
            "documents_excluded": int(documents_excluded_by.total()),
            "documents_excluded_by": dict(sorted(documents_excluded_by.items())),
            "documents_selected": int(documents_selected),
            "quota_boundary_prefix_truncations": int(prefix_truncations),
            "tokens_by": _normalize_counter_tree(tokens_by),
            "documents_by": _normalize_counter_tree(documents_by),
            "shards": committed_inventory,
            "elapsed_seconds": float(elapsed_before + max(time.time() - started, 0.0)),
        }
        write_json_atomic(
            staging / "build_state.json",
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        files_since_checkpoint = 0

    try:
        source_names = sorted({key.partition("|")[0] for key in quotas})
        for source in source_names:
            source_files = list(files_by_source.get(source, {}).get(split, []))
            if not source_files:
                raise ValueError(f"cleaning report has no {split} files for {source}")
            random.Random(_stable_source_seed(seed, split, source)).shuffle(
                source_files
            )
            required_columns = {
                "text",
                "source",
                "source_revision",
                "repo_id",
                "license",
                "language",
                "domain",
                "length_bucket",
                "mix_bucket",
                "document_sha256",
                "document_timestamp",
            } | document_admission.required_columns
            for path in source_files:
                path_key = str(path.resolve())
                if path_key in processed_set:
                    continue
                source_remaining = sum(
                    value
                    for key, value in remaining.items()
                    if key.partition("|")[0] == source
                )
                if source_remaining <= 0:
                    break
                parquet = pq.ParquetFile(path)
                missing = required_columns - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(
                        f"shard input is missing {sorted(missing)}: {path}"
                    )
                for batch in parquet.iter_batches(
                    batch_size=max(int(batch_size), 1),
                    columns=sorted(required_columns),
                ):
                    rows = batch.to_pylist()
                    documents_read += len(rows)
                    rows = _admitted_rows(
                        rows,
                        source=source,
                        split=split,
                        path=path,
                        document_admission=document_admission,
                        exclusions=documents_excluded_by,
                    )
                    if not rows:
                        continue
                    encoded = _tokenize_texts(
                        tokenizer,
                        [str(row.get("text") or "") for row in rows],
                    )
                    if len(encoded) != len(rows):
                        raise RuntimeError(
                            "tokenizer output count does not match shard input"
                        )
                    for row, ids in zip(rows, encoded, strict=True):
                        bucket = str(row.get("mix_bucket") or "")
                        available = int(remaining.get(bucket, 0))
                        if available <= 0 or not ids:
                            continue
                        token_array, truncated = _accepted_tokens(
                            ids,
                            eos_id=int(tokenizer.eos_token_id),
                            remaining=available,
                        )
                        accepted = int(token_array.size)
                        if accepted <= 0:
                            continue
                        writer.write(token_array)
                        remaining[bucket] -= accepted
                        documents_selected += 1
                        prefix_truncations += int(truncated)
                        dimensions = _row_dimensions(
                            row,
                            token_count=len(ids) + 1,
                        )
                        for dimension, value in dimensions.items():
                            tokens_by[dimension][value] += accepted
                            documents_by[dimension][value] += 1
                    if not any(
                        value > 0
                        for key, value in remaining.items()
                        if key.partition("|")[0] == source
                    ):
                        break
                processed_files.append(path_key)
                processed_set.add(path_key)
                files_since_checkpoint += 1
                if files_since_checkpoint >= int(checkpoint_every_files):
                    checkpoint()
            if files_since_checkpoint > 0:
                checkpoint()
            unmet_source = {
                key: value
                for key, value in remaining.items()
                if key.partition("|")[0] == source and value > 0
            }
            if unmet_source:
                raise RuntimeError(
                    f"unique clean supply did not fill quotas: {unmet_source}"
                )
        unmet = {key: value for key, value in remaining.items() if value > 0}
        if unmet:
            raise RuntimeError(f"unique clean supply did not fill quotas: {unmet}")
        checkpoint(status="complete")
        if not writer.shards:
            raise RuntimeError(f"shard build emitted no tokens for split={split}")
        manifest = TokenShardManifest(
            dtype="int32",
            shards=tuple(writer.shards),
            eos_token_id=int(tokenizer.eos_token_id),
            tokenizer_sha1=tokenizer_sha1,
        )
        manifest_path = staging / "manifest.json"
        save_manifest(str(manifest_path), manifest)
        committed_by_path = {str(item["path"]): item for item in committed_inventory}
        shard_files = []
        for shard in manifest.shards:
            path = staging / shard.path
            committed = committed_by_path.get(shard.path)
            if (
                committed is None
                or int(committed.get("tokens") or 0) != int(shard.tokens)
                or int(committed.get("bytes") or 0) != int(path.stat().st_size)
                or not str(committed.get("sha256") or "")
            ):
                raise RuntimeError(
                    f"completed shard is missing from committed inventory: {path}"
                )
            shard_files.append(
                {
                    "path": shard.path,
                    "tokens": int(shard.tokens),
                    "bytes": int(path.stat().st_size),
                    "sha256": str(committed["sha256"]),
                }
            )
        report = {
            "schema": SPLIT_REPORT_SCHEMA,
            "status": "complete",
            "split": split,
            "seed": int(seed),
            "tokenizer_bundle_sha1": tokenizer_sha1,
            "target_tokens": int(sum(quotas.values())),
            "actual_tokens": int(manifest.total_tokens),
            "target_tokens_by_mix_bucket": dict(sorted(quotas.items())),
            "documents_read": int(documents_read),
            "documents_excluded": int(documents_excluded_by.total()),
            "documents_excluded_by": dict(sorted(documents_excluded_by.items())),
            "documents_selected": int(documents_selected),
            "quota_boundary_prefix_truncations": int(prefix_truncations),
            "tokens_by": _normalize_counter_tree(tokens_by),
            "documents_by": _normalize_counter_tree(documents_by),
            "lineage": dict(fingerprints),
            "manifest_sha1": sha1_file(str(manifest_path)),
            "shards": shard_files,
            "duration_seconds": float(elapsed_before + max(time.time() - started, 0.0)),
        }
        if report["actual_tokens"] != report["target_tokens"]:
            raise RuntimeError("split shard token count does not match exact quota")
        write_json_atomic(
            staging / "build_report.json",
            report,
            ensure_ascii=False,
            sort_keys=True,
        )
        _write_output_dir_marker(str(staging))
        _promote_staging_out_dir(
            staging_out_dir=str(staging),
            final_out_dir=str(output_dir),
            overwrite_output_dir=False,
        )
        return report
    except BaseException:
        writer.close()
        raise


def _root_marker_payload(
    *, tokenizer_sha1: str, fingerprints: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA,
        "status": "building",
        "tokenizer_bundle_sha1": tokenizer_sha1,
        "lineage": dict(fingerprints),
    }


def _copy_lineage_file(*, source: Path, destination: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"lineage artifact is missing: {source}")
    expected_sha256 = _sha256_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256_path(destination) != expected_sha256:
            raise ValueError(
                f"existing lineage artifact fingerprint mismatch: {destination}"
            )
    else:
        temporary = destination.with_name(
            f"{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {
        "path": str(destination.relative_to(destination.parents[1])).replace("\\", "/"),
        "sha256": expected_sha256,
        "bytes": int(destination.stat().st_size),
    }


def _materialize_lineage_artifacts(
    *,
    output_root: Path,
    clean_root: Path,
    cleaning_report_path: Path,
    cleaning: dict[str, Any],
    profile_path: str,
    mix_plan_path: str,
    admission_policy: str,
) -> dict[str, dict[str, object]]:
    lineage_root = output_root / "lineage"
    sources: list[tuple[str, Path, str]] = [
        ("admission_policy", Path(admission_policy), "admission_policy.json"),
        (
            "cleaning_report",
            cleaning_report_path,
            "cleaning_report.json",
        ),
        ("token_profile", Path(profile_path), "token_profile.json"),
        ("resolved_mix_plan", Path(mix_plan_path), "resolved_mix_plan.json"),
    ]
    mix = _load_json(Path(mix_plan_path))
    mix_policy_value = str(mix.get("mix_policy_path") or "").strip()
    if not mix_policy_value:
        raise ValueError("resolved mix has no pinned mix policy")
    sources.append(("mix_policy", Path(mix_policy_value), "mix_policy.json"))
    acquisition_value = str(cleaning.get("acquisition_report") or "").strip()
    if acquisition_value:
        acquisition_path = Path(acquisition_value).expanduser().resolve()
        sources.append(
            ("acquisition_report", acquisition_path, "acquisition_report.json")
        )
        acquisition = _load_json(acquisition_path)
        inventory_value = str(acquisition.get("inventory_path") or "").strip()
        if inventory_value:
            inventory_path = Path(inventory_value).expanduser().resolve()
            sources.append(
                ("source_inventory", inventory_path, "source_inventory.json")
            )
            inventory = _load_json(inventory_path)
            selection_value = str(inventory.get("selection_path") or "").strip()
            if selection_value:
                selection_path = Path(selection_value).expanduser().resolve()
                recorded_selection_sha256 = str(inventory.get("selection_sha256") or "")
                if _sha256_path(selection_path) != recorded_selection_sha256:
                    raise ValueError("source inventory selection fingerprint mismatch")
                sources.append(
                    ("source_selection", selection_path, "source_selection.json")
                )
                selection = _load_json(selection_path)
                audit_value = str(selection.get("source_audit_path") or "").strip()
                if audit_value:
                    audit_path = Path(audit_value).expanduser().resolve()
                    recorded_audit_sha256 = str(
                        selection.get("source_audit_sha256") or ""
                    )
                    if _sha256_path(audit_path) != recorded_audit_sha256:
                        raise ValueError("source selection audit fingerprint mismatch")
                    sources.append(("source_audit", audit_path, "source_audit.json"))
                    audit = _load_json(audit_path)
                    sample_report = audit.get("distributed_sample_report")
                    if isinstance(sample_report, dict):
                        sample_value = str(sample_report.get("path") or "").strip()
                        sample_sha256 = str(sample_report.get("sha256") or "")
                        if sample_value:
                            sample_path = Path(sample_value).expanduser().resolve()
                            if _sha256_path(sample_path) != sample_sha256:
                                raise ValueError(
                                    "source audit distributed sample fingerprint mismatch"
                                )
                            sources.append(
                                (
                                    "source_distributed_sample_report",
                                    sample_path,
                                    "source_distributed_sample_report.json",
                                )
                            )
                    candidate_cards = audit.get("pinned_candidate_cards", [])
                    if not isinstance(candidate_cards, list):
                        raise ValueError("source audit candidate cards must be a list")
                    for card_index, card in enumerate(candidate_cards):
                        if not isinstance(card, dict):
                            raise ValueError(
                                "source audit candidate cards must be objects"
                            )
                        card_value = str(card.get("path") or "").strip()
                        card_sha256 = str(card.get("sha256") or "")
                        if not card_value:
                            raise ValueError("source audit candidate card has no path")
                        card_path = Path(card_value).expanduser().resolve()
                        if _sha256_path(card_path) != card_sha256:
                            raise ValueError(
                                "source audit candidate card fingerprint mismatch"
                            )
                        sources.append(
                            (
                                f"source_candidate_card_{card_index:03d}",
                                card_path,
                                f"source_candidate_card_{card_index:03d}.md",
                            )
                        )
    for index, value in enumerate(cleaning.get("benchmark_jsonl", [])):
        benchmark_path = Path(str(value)).expanduser().resolve()
        sources.append(
            (
                f"benchmark_signatures_{index:03d}",
                benchmark_path,
                f"benchmark_signatures_{index:03d}_{benchmark_path.name}",
            )
        )
    exclusion_value = str(cleaning.get("legacy_content_exclusion_report") or "").strip()
    if exclusion_value:
        sources.append(
            (
                "legacy_content_exclusion_report",
                Path(exclusion_value).expanduser().resolve(),
                "legacy_content_exclusion_report.json",
            )
        )
    quality_proxy_value = str(
        cleaning.get("chinese_quality_proxy_report") or ""
    ).strip()
    if quality_proxy_value:
        sources.append(
            (
                "chinese_quality_proxy_report",
                Path(quality_proxy_value).expanduser().resolve(),
                "chinese_quality_proxy_report.json",
            )
        )
    quality_exemptions = cleaning.get(
        "chinese_quality_proxy_curated_source_exemptions", []
    )
    if not isinstance(quality_exemptions, list):
        raise ValueError("cleaning quality proxy curated exemptions must be a list")
    for index, exemption in enumerate(quality_exemptions):
        if not isinstance(exemption, dict):
            raise ValueError(
                "cleaning quality proxy curated exemptions must be objects"
            )
        evidence_value = str(exemption.get("evidence_report_path") or "").strip()
        if not evidence_value:
            raise ValueError("quality proxy curated exemption has no evidence report")
        sources.append(
            (
                f"chinese_quality_proxy_exemption_{index:03d}",
                Path(evidence_value).expanduser().resolve(),
                f"chinese_quality_proxy_exemption_{index:03d}.json",
            )
        )
    artifacts: dict[str, dict[str, object]] = {}
    for label, source, filename in sources:
        artifacts[label] = _copy_lineage_file(
            source=source,
            destination=lineage_root / filename,
        )
    return artifacts


def _matching_root_marker(
    *, existing: dict[str, Any], expected: dict[str, Any]
) -> bool:
    return all(
        key == "status" or existing.get(key) == value for key, value in expected.items()
    ) and str(existing.get("status") or "") in {"building", "complete"}


def _validate_existing_split(
    *, split_dir: Path, report: dict[str, Any], quotas: dict[str, int]
) -> None:
    if (
        str(report.get("schema")) != SPLIT_REPORT_SCHEMA
        or str(report.get("status")) != "complete"
        or report.get("target_tokens_by_mix_bucket") != dict(sorted(quotas.items()))
    ):
        raise ValueError(f"existing split is not resumable: {split_dir}")
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file() or sha1_file(str(manifest_path)) != str(
        report.get("manifest_sha1") or ""
    ):
        raise ValueError(f"existing split manifest fingerprint mismatch: {split_dir}")
    shards = report.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"existing split has no shard inventory: {split_dir}")
    for item in shards:
        if not isinstance(item, dict):
            raise ValueError(f"invalid existing shard inventory: {split_dir}")
        path = (split_dir / str(item.get("path") or "")).resolve()
        if path.parent != split_dir.resolve() or not path.is_file():
            raise ValueError(f"existing split shard is missing: {path}")
        if int(path.stat().st_size) != int(item.get("bytes") or 0):
            raise ValueError(f"existing split shard size mismatch: {path}")
        if sha256_file(path) != str(item.get("sha256") or ""):
            raise ValueError(f"existing split shard fingerprint mismatch: {path}")


def build_pretrain_token_dataset(
    *,
    clean_root: str,
    cleaning_report_path: str = "",
    profile_path: str,
    mix_plan_path: str,
    output_root: str,
    tokenizer_path: str = "ml/modeling/text",
    admission_policy: str = "configs/data/pretrain_data_admission_policy.json",
    shard_size_tokens: int = 50_000_000,
    batch_size: int = 64,
    checkpoint_every_files: int = 25,
    seed: int = 20260719,
) -> dict[str, Any]:
    (
        clean,
        cleaning,
        profile,
        mix,
        tokenizer_sha1,
        fingerprints,
    ) = _validate_inputs(
        clean_root=clean_root,
        cleaning_report_path=cleaning_report_path,
        profile_path=profile_path,
        mix_plan_path=mix_plan_path,
        tokenizer_path=tokenizer_path,
        admission_policy=admission_policy,
    )
    policy_path = Path(admission_policy).expanduser().resolve()
    policy = _load_json(policy_path)
    document_admission = load_post_clean_document_admission(
        policy,
        policy_path=policy_path,
    )
    output = Path(output_root).expanduser().resolve()
    marker_payload = _root_marker_payload(
        tokenizer_sha1=tokenizer_sha1,
        fingerprints=fingerprints,
    )
    marker_path = output / ROOT_MARKER
    if output.exists() and any(output.iterdir()):
        if not marker_path.is_file() or not _matching_root_marker(
            existing=_load_json(marker_path),
            expected=marker_payload,
        ):
            raise ValueError(
                f"output_root is not the matching resumable shard build: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(marker_path, marker_payload, sort_keys=True)

    lineage_artifacts = _materialize_lineage_artifacts(
        output_root=output,
        clean_root=clean,
        cleaning_report_path=Path(
            str(cleaning_report_path or clean / "cleaning_report.json")
        )
        .expanduser()
        .resolve(),
        cleaning=cleaning,
        profile_path=profile_path,
        mix_plan_path=mix_plan_path,
        admission_policy=admission_policy,
    )

    tokenizer = _load_tokenizer(str(Path(tokenizer_path).expanduser().resolve()))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_id, int):
        raise ValueError("selected tokenizer has no eos_token_id")
    files_by_source = _files_by_source_split(clean_root=clean, cleaning=cleaning)
    split_reports: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        quotas = _split_quotas(split=split, profile=profile, mix=mix)
        split_dir = output / split
        report_path = split_dir / "build_report.json"
        if report_path.is_file():
            report = _load_json(report_path)
            if report.get("lineage") != fingerprints:
                raise ValueError(f"existing split is not resumable: {split_dir}")
            _validate_existing_split(
                split_dir=split_dir,
                report=report,
                quotas=quotas,
            )
        else:
            report = _write_split(
                split=split,
                files_by_source=files_by_source,
                quotas=quotas,
                tokenizer=tokenizer,
                tokenizer_sha1=tokenizer_sha1,
                output_dir=split_dir,
                shard_size_tokens=int(shard_size_tokens),
                batch_size=int(batch_size),
                checkpoint_every_files=int(checkpoint_every_files),
                seed=int(seed),
                fingerprints=fingerprints,
                document_admission=document_admission,
            )
        split_reports[split] = report

    dataset_manifest = {
        "schema": DATASET_SCHEMA,
        "status": "complete",
        "clean_root": str(clean),
        "output_root": str(output),
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "tokenizer_bundle_sha1": tokenizer_sha1,
        "lineage": fingerprints,
        "lineage_artifacts": lineage_artifacts,
        "unique_document_policy": "clean corpus global exact/near dedup; each split scanned once",
        "splits": {
            split: {
                "manifest": f"{split}/manifest.json",
                "manifest_sha1": str(report["manifest_sha1"]),
                "tokens": int(report["actual_tokens"]),
                "documents_selected": int(report["documents_selected"]),
                "documents_excluded": int(report["documents_excluded"]),
                "documents_excluded_by": report["documents_excluded_by"],
                "quota_boundary_prefix_truncations": int(
                    report["quota_boundary_prefix_truncations"]
                ),
                "tokens_by": report["tokens_by"],
                "documents_by": report["documents_by"],
                "shards": report["shards"],
            }
            for split, report in split_reports.items()
        },
    }
    write_json_atomic(
        output / "dataset_manifest.json",
        dataset_manifest,
        ensure_ascii=False,
        sort_keys=True,
    )
    marker_payload["status"] = "complete"
    marker_payload["dataset_manifest_sha256"] = _sha256_path(
        output / "dataset_manifest.json"
    )
    write_json_atomic(marker_path, marker_payload, sort_keys=True)
    return dataset_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build provenance-complete unique pretraining token shards."
    )
    parser.add_argument("--clean-root", required=True)
    parser.add_argument(
        "--cleaning-report",
        default="",
        help="Immutable cleaning report snapshot for a resumable shard build.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mix-plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tokenizer", default="ml/modeling/text")
    parser.add_argument(
        "--admission-policy",
        default="configs/data/pretrain_data_admission_policy.json",
    )
    parser.add_argument("--shard-size-tokens", type=int, default=50_000_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every-files", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_pretrain_token_dataset(
        clean_root=str(args.clean_root),
        cleaning_report_path=str(args.cleaning_report),
        profile_path=str(args.profile),
        mix_plan_path=str(args.mix_plan),
        output_root=str(args.output_root),
        tokenizer_path=str(args.tokenizer),
        admission_policy=str(args.admission_policy),
        shard_size_tokens=int(args.shard_size_tokens),
        batch_size=int(args.batch_size),
        checkpoint_every_files=int(args.checkpoint_every_files),
        seed=int(args.seed),
    )
    print(
        f"[DONE] train_tokens={report['splits']['train']['tokens']:,} "
        f"output={Path(args.output_root).expanduser().resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
