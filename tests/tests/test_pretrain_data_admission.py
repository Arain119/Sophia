from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.errors import SophiaUsageError
from ml.training.pretrain import data_admission as admission_mod
from ml.training.pretrain.data_admission import (
    validate_pretrain_data_admission,
    validate_pretrain_dataset_manifest,
)


def _write_policy(path: Path, forbidden_root: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "sophia_pretrain_data_admission_policy_v1",
                "forbidden_input_roots": [str(forbidden_root)],
                "allow_legacy_assets_as_training_input": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_pretrain_data_admission_rejects_legacy_split(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy_tokens"
    policy = _write_policy(tmp_path / "policy.json", legacy_root)

    with pytest.raises(SophiaUsageError, match="legacy pretraining data is forbidden"):
        validate_pretrain_data_admission(
            data_path=str(legacy_root / "train"),
            policy_path=policy,
        )


def test_pretrain_data_admission_accepts_fresh_paths(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path / "policy.json", tmp_path / "legacy_tokens")

    validate_pretrain_data_admission(
        data_path=str(tmp_path / "fresh_tokens" / "train"),
        policy_path=policy,
    )


def test_dataset_manifest_validation_allows_canonical_pretrain_root(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "dataset" / "pretrain"
    canonical_root.mkdir(parents=True)
    policy = _write_policy(tmp_path / "policy.json", canonical_root)

    with pytest.raises(
        SophiaUsageError,
        match="production pretraining dataset manifest is missing",
    ):
        validate_pretrain_dataset_manifest(
            data_path=str(canonical_root),
            policy_path=policy,
        )


def test_pretrain_data_admission_resolves_symlinks(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy_tokens"
    legacy_root.mkdir()
    alias = tmp_path / "apparently_fresh"
    alias.symlink_to(legacy_root, target_is_directory=True)
    policy = _write_policy(tmp_path / "policy.json", legacy_root)

    with pytest.raises(SophiaUsageError, match="legacy pretraining data is forbidden"):
        validate_pretrain_data_admission(
            data_path=str(alias / "train"),
            policy_path=policy,
        )


def test_quality_proxy_lineage_binds_policy_cleaning_and_pass_report(
    tmp_path: Path,
) -> None:
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    quality = lineage / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "schema": "sophia_chinese_quality_proxy_report_v2",
                "status": "pass",
                "gate_results": {"macro_f1": True, "precision": True},
                "model": {"threshold": 0.68, "sha256": "model-hash"},
                "source": {"labels_sha256": "labels-hash"},
            }
        ),
        encoding="utf-8",
    )
    quality_sha256 = hashlib.sha256(quality.read_bytes()).hexdigest()
    cleaning = lineage / "cleaning.json"
    cleaning.write_text(
        json.dumps(
            {
                "chinese_quality_proxy_report_sha256": quality_sha256,
                "chinese_quality_proxy_model_sha256": "model-hash",
                "chinese_quality_proxy_labels_sha256": "labels-hash",
                "chinese_quality_proxy_minimum_score": 0.68,
                "chinese_quality_proxy_minimum_cjk_ratio": 0.25,
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        "chinese_quality_proxy_report": {
            "path": "lineage/quality.json",
            "sha256": quality_sha256,
        },
        "cleaning_report": {"path": "lineage/cleaning.json"},
    }
    policy = {
        "chinese_quality_proxy": {
            "required": True,
            "minimum_score": 0.68,
            "minimum_cjk_ratio": 0.25,
        }
    }

    admission_mod._validate_chinese_quality_proxy_lineage(
        root=tmp_path,
        policy=policy,
        artifacts=artifacts,
    )
    policy["chinese_quality_proxy"]["minimum_score"] = 0.70
    with pytest.raises(SophiaUsageError, match="thresholds do not match"):
        admission_mod._validate_chinese_quality_proxy_lineage(
            root=tmp_path,
            policy=policy,
            artifacts=artifacts,
        )


def test_quality_proxy_lineage_binds_curated_exemption_evidence(
    tmp_path: Path,
) -> None:
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    quality = lineage / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "schema": "sophia_chinese_quality_proxy_report_v2",
                "status": "pass",
                "gate_results": {"test": True},
                "model": {"threshold": 0.68, "sha256": "model-hash"},
                "source": {"labels_sha256": "labels-hash"},
            }
        ),
        encoding="utf-8",
    )
    quality_sha256 = hashlib.sha256(quality.read_bytes()).hexdigest()
    evidence = lineage / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "sophia_curated_chinese_source_proxy_exemption_v1",
                "status": "pass",
                "source": {"repo_id": "owner/repo", "revision": "a" * 40},
                "gate_results": {"sample": True, "rights": True},
            }
        ),
        encoding="utf-8",
    )
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    exemption = {
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "reason": "curated source",
        "evidence_report_sha256": evidence_sha256,
    }
    cleaning = lineage / "cleaning.json"
    cleaning.write_text(
        json.dumps(
            {
                "chinese_quality_proxy_report_sha256": quality_sha256,
                "chinese_quality_proxy_model_sha256": "model-hash",
                "chinese_quality_proxy_labels_sha256": "labels-hash",
                "chinese_quality_proxy_minimum_score": 0.68,
                "chinese_quality_proxy_minimum_cjk_ratio": 0.25,
                "chinese_quality_proxy_curated_source_exemptions": [exemption],
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        "chinese_quality_proxy_report": {
            "path": "lineage/quality.json",
            "sha256": quality_sha256,
        },
        "chinese_quality_proxy_exemption_000": {
            "path": "lineage/evidence.json",
            "sha256": evidence_sha256,
        },
        "cleaning_report": {"path": "lineage/cleaning.json"},
    }
    policy = {
        "chinese_quality_proxy": {
            "required": True,
            "minimum_score": 0.68,
            "minimum_cjk_ratio": 0.25,
            "curated_source_exemptions": [exemption],
        }
    }

    admission_mod._validate_chinese_quality_proxy_lineage(
        root=tmp_path,
        policy=policy,
        artifacts=artifacts,
    )
    artifacts["chinese_quality_proxy_exemption_000"]["sha256"] = "0" * 64
    with pytest.raises(SophiaUsageError, match="evidence is invalid"):
        admission_mod._validate_chinese_quality_proxy_lineage(
            root=tmp_path,
            policy=policy,
            artifacts=artifacts,
        )
