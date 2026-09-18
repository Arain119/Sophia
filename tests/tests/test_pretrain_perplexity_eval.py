from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ml.tooling.scripts import eval_pretrain_perplexity as mod


def test_release_binding_hashes_files_and_aggregates_dataset_fingerprints(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    checkpoint = tmp_path / "checkpoint.bin"
    model = tmp_path / "model.bin"
    protocol.write_bytes(b"protocol")
    checkpoint.write_bytes(b"checkpoint")
    model.write_bytes(b"model")
    binding = mod._evaluation_binding(
        protocol_json=protocol,
        checkpoint=checkpoint,
        checkpoint_sha256=None,
        eval_export_model=model,
        dataset_fingerprints={"zh": "z", "en": "e"},
    )
    aggregate = hashlib.sha256(b"en=e\nzh=z\n").hexdigest()
    assert binding["dataset_fingerprints"] == {"en": "e", "zh": "z"}
    assert binding["evaluation_asset_sha256"] == aggregate


def test_parse_dataset_specs() -> None:
    assert mod.parse_dataset_specs(["target=target/manifest.json", "old=old.json"]) == {
        "target": "target/manifest.json",
        "old": "old.json",
    }


def test_parse_dataset_specs_rejects_duplicate_name() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        mod.parse_dataset_specs(["target=one.json", "target=two.json"])
