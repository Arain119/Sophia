from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rtx5090_bf16_sdpa_flash_liger_inductor_recipe_is_pinned() -> None:
    recipe_path = (
        REPO_ROOT
        / "configs"
        / "pretrain"
        / "machine_recipes"
        / "rtx5090_bf16_seq4096_bs3_acc22.json"
    )

    payload = json.loads(recipe_path.read_text(encoding="utf-8"))

    assert payload["selected_name"] == (
        "release_pretrain_rtx5090_bf16_sdpa_flash_liger_inductor"
    )
    assert float(payload["release_semantics"]["learning_rate"]) == 1e-4
    assert float(payload["release_semantics"]["adam_eps"]) == 1e-6
    assert float(payload["release_semantics"]["embedding_lr_scale"]) == 0.25
    assert float(payload["release_semantics"]["ema_decay"]) == 0.0
    assert int(payload["release_semantics"]["warmup_steps"]) == 800
    assert int(payload["release_semantics"]["target_tokens_per_update"]) == 270336
    assert payload["release_semantics"]["profile"]["curriculum"] == {
        "default_total_tokens": 0,
        "max_seq_len": 4096,
        "stage_token_weights": [250],
        "train_seq_len": 4096,
        "train_seq_stages": [4096],
    }
    assert int(payload["release_semantics"]["profile"]["model"]["max_seq_len"]) == 4096
    assert payload["machine_recipe"] == {
        "accumulation_steps": 22,
        "batch_size": 3,
        "gradient_checkpointing": 0,
        "gradient_checkpointing_exclude_first": 0,
        "gradient_checkpointing_exclude_last": 0,
        "loss_chunk_size": 0,
    }
    assert payload["machine_runtime"] == {
        "dataloader_num_workers": 0,
        "dataloader_persistent_workers": 0,
        "dataloader_prefetch_factor": 0,
        "shard_preload": 1,
        "shard_preload_bytes": 4194304,
        "step_execution_backend": "inductor",
    }
    model = payload["release_semantics"]["profile"]["model"]
    assert model["dim"] == 1536
    assert model["n_layers"] == 30
    assert model["n_heads"] == 12
    assert model["ffn_hidden"] == 4096
    assert model["num_key_value_heads"] == 4
    assert model["rope_head_dim"] == 64
    assert int(payload["evidence"]["model_params_estimate"]) == 830_573_568
    assert 19.0 < float(payload["evidence"]["tokens_per_parameter"]) < 20.5
    assert payload["evidence"]["runtime_measurement_status"] == (
        "rtx5090_bf16_bs3_acc22_validated_short_probe"
    )
    assert payload["machine_signature"]["cuda_device_name"] == "NVIDIA GeForce RTX 5090"
    assert payload["machine_signature"]["cuda_capability"] == [12, 0]
    assert float(payload["machine_signature"]["cuda_total_memory_gb"]) > 31.0
    assert payload["artifacts"]["machine_recipe"]["best"]["batch_size"] == 3
    assert payload["artifacts"]["machine_recipe"]["best"]["accumulation_steps"] == 22
    assert payload["artifacts"]["machine_recipe"]["best"]["tokens_per_sec"] > 27000.0
    assert payload["artifacts"]["machine_recipe"]["best"]["max_memory_gb"] > 27.0
    assert payload["artifacts"]["update_profile"]["tokens_per_update"] == 270336


def test_machine_recipe_directory_has_only_current_release_recipe() -> None:
    recipes_root = REPO_ROOT / "configs" / "pretrain" / "machine_recipes"

    assert [path.name for path in sorted(recipes_root.glob("*.json"))] == [
        "rtx5090_bf16_seq4096_bs3_acc22.json"
    ]


def test_all_machine_recipe_profiles_use_canonical_context() -> None:
    recipes_root = REPO_ROOT / "configs" / "pretrain" / "machine_recipes"

    for recipe_path in sorted(recipes_root.glob("*.json")):
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
        curriculum = payload["release_semantics"]["profile"]["curriculum"]
        model = payload["release_semantics"]["profile"]["model"]
        assert curriculum["max_seq_len"] == 4096, recipe_path.name
        assert curriculum["train_seq_stages"] == [4096], recipe_path.name
        assert curriculum["stage_token_weights"] == [250], recipe_path.name
        assert model["max_seq_len"] == 4096, recipe_path.name
