from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.data.token_shards.shard_manifest import load_manifest
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.tooling.scripts.data import build_pretrain_token_shards as mod
from ml.training.pretrain.data_admission import validate_pretrain_dataset_manifest
from ml.training.pretrain.shard_builder_tokenizer import _load_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = REPO_ROOT / "ml" / "modeling" / "text"
SOURCE = "fresh_source_v1"


def _document_hash(split: str, suffix: int) -> str:
    prefix = {"train": 0, "val": 9_600, "test": 9_800}[split] + int(suffix)
    return f"{prefix:08x}" + "0" * 56


def _row(split: str, index: int) -> dict[str, object]:
    text = f"{split} example {index}: 这是全新且可追踪的中英文训练文档。"
    return {
        "text": text,
        "source": SOURCE,
        "source_revision": "a" * 40,
        "repo_id": "owner/fresh-source",
        "license": "cc-by-4.0",
        "domain": "test_domain",
        "quality_score": "high",
        "language": "zh",
        "document_id": f"{split}-{index}",
        "url": "",
        "document_sha256": _document_hash(split, index),
        "near_signature": "",
        "document_timestamp": "2026-07-01T00:00:00Z",
        "source_file_sha256": "b" * 64,
        "character_count": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "length_bucket": "chars_lt_2k",
        "mix_bucket": f"{SOURCE}|chars_lt_2k",
        "cjk_ratio": 0.5,
        "latin_ratio": 0.5,
    }


def _write_inputs(tmp_path: Path) -> dict[str, Path | int]:
    clean = tmp_path / "fresh_clean"
    output_files: list[str] = []
    token_counts: dict[str, int] = {}
    tokenizer = _load_tokenizer(str(TOKENIZER))
    for split in ("train", "val", "test"):
        rows = [_row(split, index) for index in range(2)]
        for index, row in enumerate(rows):
            path = clean / SOURCE / split / "input-00000" / f"part-{index:05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([row]), path)
            output_files.append(str(path.resolve()))
        token_counts[split] = sum(
            len(item.ids) + 1
            for item in tokenizer.encode_batch([str(row["text"]) for row in rows])
        )
    cleaning = {
        "schema": "sophia_clean_pretrain_corpus_v1",
        "status": "complete",
        "output_root": str(clean.resolve()),
        "sources": [
            {
                "name": SOURCE,
                "output_files": output_files,
                "output_file_inventory": [
                    {
                        "path": path,
                        "bytes": Path(path).stat().st_size,
                        "sha256": mod.sha256_file(path),
                    }
                    for path in output_files
                ],
            }
        ],
    }
    cleaning_path = clean / "cleaning_report.json"
    cleaning_path.write_text(json.dumps(cleaning), encoding="utf-8")

    tokenizer_sha1 = compute_tokenizer_bundle_sha1(str(TOKENIZER))
    policy = {
        "schema": "sophia_pretrain_data_admission_policy_v1",
        "forbidden_input_roots": [str(tmp_path / "legacy")],
        "allow_legacy_assets_as_training_input": False,
        "required_tokenizer_bundle_sha1": tokenizer_sha1,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    bucket = f"{SOURCE}|chars_lt_2k"
    profile = {
        "schema": "sophia_pretrain_corpus_token_profile_v1",
        "status": "complete",
        "clean_root": str(clean.resolve()),
        "cleaning_report": str(cleaning_path.resolve()),
        "cleaning_report_sha256": mod._sha256_path(cleaning_path),
        "admission_policy": str(policy_path.resolve()),
        "admission_policy_sha256": mod._sha256_path(policy_path),
        "tokenizer_bundle_sha1": tokenizer_sha1,
        "tokens_by": {
            "mix_bucket_split": {
                f"{bucket}|{split}": count for split, count in token_counts.items()
            }
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    mix_policy_path = tmp_path / "mix_policy.json"
    mix_policy_path.write_text(
        json.dumps({"schema": "sophia_pretrain_mix_policy_v1"}),
        encoding="utf-8",
    )

    train_tokens = token_counts["train"]
    mix = {
        "schema": "sophia_resolved_pretrain_mix_plan_v1",
        "status": "ready",
        "requested_unique_train_tokens": train_tokens,
        "resolved_unique_train_tokens": train_tokens,
        "tokenizer_bundle_sha1": tokenizer_sha1,
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": mod._sha256_path(profile_path),
        "mix_policy_path": str(mix_policy_path.resolve()),
        "mix_policy_sha256": mod._sha256_path(mix_policy_path),
        "source_train_quotas": {SOURCE: train_tokens},
        "composite_source_character_length_quotas": {bucket: train_tokens},
    }
    mix_path = tmp_path / "mix.json"
    mix_path.write_text(json.dumps(mix), encoding="utf-8")
    return {
        "clean": clean,
        "policy": policy_path,
        "profile": profile_path,
        "mix": mix_path,
        "mix_policy": mix_policy_path,
        "train_tokens": train_tokens,
        "val_tokens": token_counts["val"],
        "test_tokens": token_counts["test"],
    }


def _refresh_lineage(inputs: dict[str, Path | int]) -> None:
    cleaning_path = Path(inputs["clean"]) / "cleaning_report.json"
    profile_path = Path(inputs["profile"])
    policy_path = Path(inputs["policy"])
    profile = json.loads(profile_path.read_text())
    profile["cleaning_report_sha256"] = mod._sha256_path(cleaning_path)
    profile["admission_policy_sha256"] = mod._sha256_path(policy_path)
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    mix_path = Path(inputs["mix"])
    mix_policy_path = Path(inputs["mix_policy"])
    mix = json.loads(mix_path.read_text())
    mix["profile_sha256"] = mod._sha256_path(profile_path)
    mix["mix_policy_sha256"] = mod._sha256_path(mix_policy_path)
    mix_path.write_text(json.dumps(mix), encoding="utf-8")


def _build(tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, Path | int]]:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "tokens"
    report = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(output),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
        shard_size_tokens=1_024,
        batch_size=2,
    )
    return report, output, inputs


def test_build_pretrain_token_dataset_writes_auditable_three_split_manifest(
    tmp_path: Path,
) -> None:
    report, output, inputs = _build(tmp_path)

    assert report["status"] == "complete"
    for split in ("train", "val", "test"):
        split_report = report["splits"][split]
        assert split_report["tokens"] == inputs[f"{split}_tokens"]
        manifest = load_manifest(str(output / split / "manifest.json"))
        assert manifest.total_tokens == inputs[f"{split}_tokens"]
        assert split_report["tokens_by"]["source"] == {
            SOURCE: inputs[f"{split}_tokens"]
        }
        assert split_report["tokens_by"]["document_time_year"] == {
            "2026": inputs[f"{split}_tokens"]
        }
        assert split_report["tokens_by"]["source_document_time_year"] == {
            f"{SOURCE}|2026": inputs[f"{split}_tokens"]
        }
        assert split_report["shards"][0]["sha256"]
    assert (output / "dataset_manifest.json").is_file()
    assert json.loads((output / mod.ROOT_MARKER).read_text())["status"] == "complete"
    assert "mix_policy" in report["lineage_artifacts"]
    validate_pretrain_dataset_manifest(
        data_path=str(output / "train"),
        tokenizer_path=str(TOKENIZER),
        policy_path=str(inputs["policy"]),
    )

    resumed = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(output),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
        shard_size_tokens=1_024,
        batch_size=2,
    )
    assert resumed["splits"]["train"]["tokens"] == inputs["train_tokens"]


def test_shard_build_applies_same_post_clean_document_admission(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _write_inputs(tmp_path)
    tokenizer = _load_tokenizer(str(TOKENIZER))
    one_document_tokens = {
        split: len(tokenizer.encode_batch([str(_row(split, 1)["text"])])[0].ids) + 1
        for split in ("train", "val", "test")
    }
    profile_path = Path(inputs["profile"])
    profile = json.loads(profile_path.read_text())
    bucket = f"{SOURCE}|chars_lt_2k"
    profile["tokens_by"]["mix_bucket_split"] = {
        f"{bucket}|{split}": one_document_tokens[split]
        for split in ("train", "val", "test")
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    mix_path = Path(inputs["mix"])
    mix = json.loads(mix_path.read_text())
    mix["requested_unique_train_tokens"] = one_document_tokens["train"]
    mix["resolved_unique_train_tokens"] = one_document_tokens["train"]
    mix["source_train_quotas"] = {SOURCE: one_document_tokens["train"]}
    mix["composite_source_character_length_quotas"] = {
        bucket: one_document_tokens["train"]
    }
    mix_path.write_text(json.dumps(mix), encoding="utf-8")
    _refresh_lineage(inputs)

    class FakeAdmission:
        required_columns: set[str] = set()

        @staticmethod
        def exclusion_reason(row: dict[str, object]) -> str:
            return "test_exclusion" if "example 0:" in str(row["text"]) else ""

    monkeypatch.setattr(
        mod,
        "load_post_clean_document_admission",
        lambda _policy, **_kwargs: FakeAdmission(),
    )
    monkeypatch.setattr(mod.random.Random, "shuffle", lambda _self, _items: None)

    report = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(profile_path),
        mix_plan_path=str(mix_path),
        output_root=str(tmp_path / "filtered_tokens"),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
        shard_size_tokens=1_024,
        batch_size=2,
    )

    for split in ("train", "val", "test"):
        assert report["splits"][split]["documents_excluded"] == 1
        assert report["splits"][split]["documents_excluded_by"] == {"test_exclusion": 1}
        assert report["splits"][split]["documents_selected"] == 1


def test_build_pretrain_token_dataset_resumes_from_committed_file_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    interrupted_output = tmp_path / "tokens_interrupted"
    original_write = mod.ShardWriter.write
    writes = 0

    def _interrupt_after_second_file_write(writer, tokens):
        nonlocal writes
        writes += 1
        original_write(writer, tokens)
        if writes == 2:
            raise KeyboardInterrupt("fixture interruption")

    monkeypatch.setattr(mod.ShardWriter, "write", _interrupt_after_second_file_write)
    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        mod.build_pretrain_token_dataset(
            clean_root=str(inputs["clean"]),
            profile_path=str(inputs["profile"]),
            mix_plan_path=str(inputs["mix"]),
            output_root=str(interrupted_output),
            tokenizer_path=str(TOKENIZER),
            admission_policy=str(inputs["policy"]),
            shard_size_tokens=1_024,
            batch_size=1,
            checkpoint_every_files=1,
        )

    staging = interrupted_output / "train.building"
    state = json.loads((staging / "build_state.json").read_text())
    assert len(state["processed_files"]) == 1
    assert len(state["shards"]) == 1
    committed_sha256 = state["shards"][0]["sha256"]
    assert (staging / "shard_00001.bin").is_file()

    monkeypatch.setattr(mod.ShardWriter, "write", original_write)
    resumed = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(interrupted_output),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
        shard_size_tokens=1_024,
        batch_size=1,
        checkpoint_every_files=1,
    )
    uninterrupted = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(tmp_path / "tokens_uninterrupted"),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
        shard_size_tokens=1_024,
        batch_size=1,
        checkpoint_every_files=1,
    )

    resumed_shards = resumed["splits"]["train"]["shards"]
    uninterrupted_shards = uninterrupted["splits"]["train"]["shards"]
    assert resumed_shards == uninterrupted_shards
    assert resumed_shards[0]["sha256"] == committed_sha256
    assert (
        resumed["splits"]["train"]["tokens_by"]
        == uninterrupted["splits"]["train"]["tokens_by"]
    )


def test_build_pretrain_token_dataset_rejects_incomplete_unique_mix(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    mix = json.loads(Path(inputs["mix"]).read_text())
    mix["status"] = "insufficient_unique_supply"
    Path(inputs["mix"]).write_text(json.dumps(mix), encoding="utf-8")

    with pytest.raises(ValueError, match="ready resolved mix plan"):
        mod.build_pretrain_token_dataset(
            clean_root=str(inputs["clean"]),
            profile_path=str(inputs["profile"]),
            mix_plan_path=str(inputs["mix"]),
            output_root=str(tmp_path / "tokens"),
            tokenizer_path=str(TOKENIZER),
            admission_policy=str(inputs["policy"]),
        )


def test_build_rejects_profile_and_mix_lineage_drift(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    cleaning_path = Path(inputs["clean"]) / "cleaning_report.json"
    cleaning = json.loads(cleaning_path.read_text())
    cleaning["drift"] = True
    cleaning_path.write_text(json.dumps(cleaning), encoding="utf-8")

    with pytest.raises(ValueError, match="cleaning report fingerprint mismatch"):
        mod.build_pretrain_token_dataset(
            clean_root=str(inputs["clean"]),
            profile_path=str(inputs["profile"]),
            mix_plan_path=str(inputs["mix"]),
            output_root=str(tmp_path / "tokens_cleaning_drift"),
            tokenizer_path=str(TOKENIZER),
            admission_policy=str(inputs["policy"]),
        )

    _refresh_lineage(inputs)
    profile_path = Path(inputs["profile"])
    profile = json.loads(profile_path.read_text())
    profile["drift"] = True
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="token profile fingerprint mismatch"):
        mod.build_pretrain_token_dataset(
            clean_root=str(inputs["clean"]),
            profile_path=str(inputs["profile"]),
            mix_plan_path=str(inputs["mix"]),
            output_root=str(tmp_path / "tokens_profile_drift"),
            tokenizer_path=str(TOKENIZER),
            admission_policy=str(inputs["policy"]),
        )


def test_build_accepts_explicit_frozen_cleaning_report_snapshot(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    clean = Path(inputs["clean"])
    live_report = clean / "cleaning_report.json"
    frozen_report = tmp_path / "frozen_cleaning_report.json"
    frozen_report.write_bytes(live_report.read_bytes())

    live = json.loads(live_report.read_text())
    live["append_only_extension"] = True
    live_report.write_text(json.dumps(live), encoding="utf-8")

    report = mod.build_pretrain_token_dataset(
        clean_root=str(clean),
        cleaning_report_path=str(frozen_report),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(tmp_path / "tokens_from_frozen_snapshot"),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
        shard_size_tokens=1_024,
        batch_size=2,
    )

    assert report["status"] == "complete"
    assert report["lineage_artifacts"]["cleaning_report"]["sha256"] == (
        mod._sha256_path(frozen_report)
    )


def test_build_rejects_clean_output_fingerprint_drift(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    clean_file = next(Path(inputs["clean"]).glob("**/*.parquet"))
    clean_file.write_bytes(clean_file.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="clean output fingerprint mismatch"):
        mod.build_pretrain_token_dataset(
            clean_root=str(inputs["clean"]),
            profile_path=str(inputs["profile"]),
            mix_plan_path=str(inputs["mix"]),
            output_root=str(tmp_path / "tokens_clean_output_drift"),
            tokenizer_path=str(TOKENIZER),
            admission_policy=str(inputs["policy"]),
        )


def test_build_copies_legacy_exclusion_report_into_lineage(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    exclusion = tmp_path / "legacy_exclusion_report.json"
    exclusion.write_text(
        json.dumps(
            {"schema": "sophia_legacy_content_exclusion_v1", "status": "complete"}
        ),
        encoding="utf-8",
    )
    cleaning_path = Path(inputs["clean"]) / "cleaning_report.json"
    cleaning = json.loads(cleaning_path.read_text())
    cleaning["legacy_content_exclusion_report"] = str(exclusion)
    cleaning_path.write_text(json.dumps(cleaning), encoding="utf-8")
    _refresh_lineage(inputs)

    output = tmp_path / "tokens"
    report = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(output),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
    )

    artifact = report["lineage_artifacts"]["legacy_content_exclusion_report"]
    assert (output / str(artifact["path"])).read_bytes() == exclusion.read_bytes()


def test_build_copies_chinese_quality_report_into_lineage(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    quality = tmp_path / "quality_proxy_report.json"
    quality.write_text(
        json.dumps(
            {"schema": "sophia_chinese_quality_proxy_report_v2", "status": "pass"}
        ),
        encoding="utf-8",
    )
    cleaning_path = Path(inputs["clean"]) / "cleaning_report.json"
    cleaning = json.loads(cleaning_path.read_text())
    cleaning["chinese_quality_proxy_report"] = str(quality)
    cleaning_path.write_text(json.dumps(cleaning), encoding="utf-8")
    _refresh_lineage(inputs)

    output = tmp_path / "tokens"
    report = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(output),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
    )

    artifact = report["lineage_artifacts"]["chinese_quality_proxy_report"]
    assert (output / str(artifact["path"])).read_bytes() == quality.read_bytes()


def test_build_copies_curated_chinese_quality_evidence_into_lineage(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    evidence = tmp_path / "quality_exemption.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "sophia_curated_chinese_source_proxy_exemption_v1",
                "status": "pass",
            }
        ),
        encoding="utf-8",
    )
    cleaning_path = Path(inputs["clean"]) / "cleaning_report.json"
    cleaning = json.loads(cleaning_path.read_text())
    cleaning["chinese_quality_proxy_curated_source_exemptions"] = [
        {"evidence_report_path": str(evidence)}
    ]
    cleaning_path.write_text(json.dumps(cleaning), encoding="utf-8")
    _refresh_lineage(inputs)

    output = tmp_path / "tokens"
    report = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(output),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
    )

    artifact = report["lineage_artifacts"]["chinese_quality_proxy_exemption_000"]
    assert (output / str(artifact["path"])).read_bytes() == evidence.read_bytes()


def test_build_copies_source_selection_and_audit_chain_into_lineage(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    sample = tmp_path / "distributed_sample_report.json"
    sample.write_text(json.dumps({"schema": "sample"}), encoding="utf-8")
    card = tmp_path / "README.md"
    card.write_text("# Pinned source card\n", encoding="utf-8")
    audit = tmp_path / "source_audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "source_audit",
                "distributed_sample_report": {
                    "path": str(sample),
                    "sha256": mod._sha256_path(sample),
                },
                "pinned_candidate_cards": [
                    {"path": str(card), "sha256": mod._sha256_path(card)}
                ],
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "source_selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "source_selection",
                "source_audit_path": str(audit),
                "source_audit_sha256": mod._sha256_path(audit),
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "source_inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "source_inventory",
                "selection_path": str(selection),
                "selection_sha256": mod._sha256_path(selection),
            }
        ),
        encoding="utf-8",
    )
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps({"inventory_path": str(inventory)}), encoding="utf-8"
    )
    cleaning_path = Path(inputs["clean"]) / "cleaning_report.json"
    cleaning = json.loads(cleaning_path.read_text())
    cleaning["acquisition_report"] = str(acquisition)
    cleaning_path.write_text(json.dumps(cleaning), encoding="utf-8")
    _refresh_lineage(inputs)

    output = tmp_path / "tokens"
    report = mod.build_pretrain_token_dataset(
        clean_root=str(inputs["clean"]),
        profile_path=str(inputs["profile"]),
        mix_plan_path=str(inputs["mix"]),
        output_root=str(output),
        tokenizer_path=str(TOKENIZER),
        admission_policy=str(inputs["policy"]),
    )

    for label, expected in (
        ("source_inventory", inventory),
        ("source_selection", selection),
        ("source_audit", audit),
        ("source_distributed_sample_report", sample),
        ("source_candidate_card_000", card),
    ):
        artifact = report["lineage_artifacts"][label]
        assert (output / str(artifact["path"])).read_bytes() == expected.read_bytes()


def test_build_pretrain_token_dataset_detects_tampered_completed_shard(
    tmp_path: Path,
) -> None:
    _report, output, inputs = _build(tmp_path)
    shard = next((output / "train").glob("shard_*.bin"))
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="shard fingerprint mismatch"):
        validate_pretrain_dataset_manifest(
            data_path=str(output),
            tokenizer_path=str(TOKENIZER),
            policy_path=str(inputs["policy"]),
        )

    with pytest.raises(ValueError, match="shard size mismatch"):
        mod.build_pretrain_token_dataset(
            clean_root=str(inputs["clean"]),
            profile_path=str(inputs["profile"]),
            mix_plan_path=str(inputs["mix"]),
            output_root=str(output),
            tokenizer_path=str(TOKENIZER),
            admission_policy=str(inputs["policy"]),
        )
