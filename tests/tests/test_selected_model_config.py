from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.core.spec import ModelSpec
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.tooling.scripts.enumerate_model_architectures import (
    count_instantiated_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_selected_model_config_matches_canonical_model_and_tokenizer() -> None:
    selected_path = REPO_ROOT / "configs" / "model" / "sophia.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    spec = ModelSpec.from_mapping(selected["model"])

    assert spec == ModelSpec.default()
    assert count_instantiated_parameters(spec) == int(
        selected["parameter_count"]["unique_parameters"]
    )
    assert selected["architecture"]["residual"] == "block_attention_residuals"
    assert selected["architecture"]["ffn"] == "dense_situ_glu"
    assert selected["architecture"]["position_encoding"] == "nope"

    shape_study_path = (
        REPO_ROOT / "configs" / "model" / "shape_study.json"
    )
    shape_study = json.loads(shape_study_path.read_text(encoding="utf-8"))
    selected_shape = shape_study["selected"]
    parameter_count = int(selected["parameter_count"]["unique_parameters"])
    target_count = int(shape_study["target_parameter_count"])
    assert int(selected_shape["unique_parameters"]) == parameter_count
    assert int(selected_shape["target_delta"]) == parameter_count - target_count
    assert float(selected_shape["target_relative_error"]) == (
        abs(parameter_count - target_count) / target_count
    )

    optimizer_path = (
        REPO_ROOT / "configs" / "pretrain" / "muon_recipe.json"
    )
    optimizer_decision = json.loads(optimizer_path.read_text(encoding="utf-8"))
    assert int(optimizer_decision["model_parameter_count"]) == parameter_count
    assert optimizer_decision["decision"]["optimizer_kind"] == "torch_muon_hybrid"
    assert optimizer_decision["decision"]["authority"] == "user_fixed"
    assert optimizer_decision["selected_semantics"]["muon_lr_scale"] == (
        "0.2 * sqrt(max(fan_out, fan_in))"
    )

    tokenizer_dir = REPO_ROOT / selected["tokenizer"]["path"]
    assert (
        compute_tokenizer_bundle_sha1(str(tokenizer_dir))
        == (selected["tokenizer"]["bundle_sha1"])
    )
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    assert (
        hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
        == (selected["tokenizer"]["tokenizer_json_sha256"])
    )
