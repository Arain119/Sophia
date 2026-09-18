import os
import subprocess
import sys
import tempfile

import pytest
import torch

from ml.integrations.adapters.hf.config import SophiaConfig


def test_write_remote_code_bundle_writes_modeling_sophia() -> None:
    from ml.integrations.export.runtime_packager import write_remote_code_bundle
    from ml.integrations.export.runtime_packager_manifest import (
        EXPORT_CANONICAL_CONFIG_FILENAME,
        EXPORT_CACHE_DECODE_FILENAME,
        EXPORT_CONFIG_PROJECTION_FILENAME,
        EXPORT_DECODER_FORWARD_FILENAME,
        EXPORT_HF_CACHE_FILENAME,
        EXPORT_HF_CONFIG_FILENAME,
        EXPORT_HF_GENERATION_FILENAME,
        EXPORT_HF_LIFECYCLE_FILENAME,
        EXPORT_HF_PROJECTION_FILENAME,
        EXPORT_HF_SUPPORT_FILENAME,
        EXPORT_MODEL_DIR_FILENAME,
        EXPORT_INPUT_MASK_FILENAME,
        EXPORT_MODELING_FILENAME,
        EXPORT_DECODER_OUTPUT_FILENAME,
        EXPORT_DECODER_RUNTIME_FILENAME,
        EXPORT_SOPHIA_DECODER_FILENAME,
        EXPORT_PRETRAINED_BUNDLE_FILENAME,
        EXPORT_RUNTIME_BACKEND_FILENAME,
        EXPORT_DECODER_HOST_FILENAME,
        EXPORT_RUNTIME_LINEAR_FILENAME,
        EXPORT_RUNTIME_FILENAME,
    )

    assert EXPORT_MODELING_FILENAME == "modeling_sophia.py"
    assert EXPORT_RUNTIME_FILENAME == "sophia_runtime.py"
    assert EXPORT_HF_SUPPORT_FILENAME == "hf_support.py"
    assert EXPORT_HF_CACHE_FILENAME == "hf_cache.py"
    assert EXPORT_HF_CONFIG_FILENAME == "hf_config.py"
    assert EXPORT_HF_GENERATION_FILENAME == "hf_generation.py"
    assert EXPORT_HF_LIFECYCLE_FILENAME == "hf_lifecycle.py"
    assert EXPORT_DECODER_FORWARD_FILENAME == "decoder_forward.py"
    assert EXPORT_CANONICAL_CONFIG_FILENAME == "canonical_config.py"

    with tempfile.TemporaryDirectory() as td:
        write_remote_code_bundle(output_dir=td)
        out = os.path.join(td, EXPORT_MODELING_FILENAME)
        assert os.path.exists(out)
        assert os.path.exists(os.path.join(td, EXPORT_RUNTIME_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_HF_SUPPORT_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_HF_CACHE_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_HF_CONFIG_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_HF_GENERATION_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_HF_LIFECYCLE_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_CONFIG_PROJECTION_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_HF_PROJECTION_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_RUNTIME_BACKEND_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_DECODER_FORWARD_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_MODEL_DIR_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_RUNTIME_LINEAR_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_CANONICAL_CONFIG_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_CACHE_DECODE_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_INPUT_MASK_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_DECODER_OUTPUT_FILENAME))
        assert os.path.exists(
            os.path.join(td, EXPORT_DECODER_RUNTIME_FILENAME)
        )
        assert os.path.exists(os.path.join(td, EXPORT_SOPHIA_DECODER_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_PRETRAINED_BUNDLE_FILENAME))
        assert os.path.exists(os.path.join(td, EXPORT_DECODER_HOST_FILENAME))
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "SophiaForCausalLM" in content


def test_ensure_auto_map_sets_expected_entries() -> None:
    from ml.integrations.adapters.hf.loaders import (
        HF_CAUSAL_LM_CLASS,
        HF_CONFIG_CLASS,
        HF_MODELING_MODULE,
        ensure_auto_map,
    )

    class DummyConfig:
        pass

    class DummyModel:
        config = DummyConfig()

    ensure_auto_map(DummyModel())
    assert DummyModel.config.architectures == [HF_CAUSAL_LM_CLASS]
    assert DummyModel.config.auto_map == {
        "AutoConfig": f"{HF_MODELING_MODULE}.{HF_CONFIG_CLASS}",
        "AutoModelForCausalLM": f"{HF_MODELING_MODULE}.{HF_CAUSAL_LM_CLASS}",
    }


def test_hf_config_metadata_marks_native_hybrid_recipe() -> None:
    cfg = SophiaConfig()

    assert cfg.model_type == "sophia_hybrid"
    assert int(cfg.num_heads) == 16
    assert int(cfg.max_position_embeddings) == 4096
    assert not hasattr(cfg, "rope_theta")
    assert not hasattr(cfg, "rope_parameters")
    assert bool(cfg.tie_word_embeddings) is True


def test_hf_model_uses_tied_word_embeddings() -> None:
    from ml.integrations.adapters.hf.model import SophiaForCausalLM

    model = SophiaForCausalLM(
        SophiaConfig(
            vocab_size=128,
            dim=64,
            n_layers=4,
            num_heads=2,
            head_dim=32,
            ffn_hidden=128,
            kda_decay_rank=16,
            kda_output_gate_rank=16,
            mla_q_rank=16,
            mla_kv_rank=16,
        )
    ).eval()

    assert model.model.output.weight is model.model.tok_embeddings.weight


def test_hf_model_preserves_depth_scaled_residual_init() -> None:
    from ml.integrations.adapters.hf.model import SophiaForCausalLM

    torch.manual_seed(0)
    cfg = SophiaConfig(
        vocab_size=128,
        dim=64,
        n_layers=4,
        num_heads=2,
        head_dim=32,
        ffn_hidden=128,
        kda_decay_rank=16,
        kda_output_gate_rank=16,
        mla_q_rank=16,
        mla_kv_rank=16,
    )
    model = SophiaForCausalLM(cfg).eval()

    residual_std = float(cfg.initializer_range) / (
        (2.0 * float(cfg.n_layers)) ** 0.5
    )
    attn_std = float(model.model.layers[0].attn.o_proj.weight.float().std(unbiased=False).item())

    assert attn_std == pytest.approx(residual_std, rel=0.4)


def test_exported_model_dir_loads_without_repo_pythonpath() -> None:
    script = """
import os
import tempfile
import torch
from transformers import AutoModelForCausalLM
from ml.integrations.adapters.hf.model import SophiaForCausalLM, SophiaConfig
from ml.integrations.export.runtime_packager import write_remote_code_bundle
from ml.integrations.adapters.hf.loaders import ensure_auto_map

cfg = SophiaConfig(
    vocab_size=128,
    dim=64,
    n_layers=4,
    num_heads=2,
    head_dim=32,
    ffn_hidden=128,
    kda_decay_rank=16,
    kda_output_gate_rank=16,
    mla_q_rank=16,
    mla_kv_rank=16,
)
model = SophiaForCausalLM(cfg).eval()
ensure_auto_map(model)

with tempfile.TemporaryDirectory() as td:
    model.save_pretrained(td, safe_serialization=True)
    write_remote_code_bundle(output_dir=td)
    env = dict(os.environ)
    env["PYTHONPATH"] = td
    env["HF_HOME"] = os.path.join(td, "_hf_home")
    env["XDG_CACHE_HOME"] = td
    code = (
        "from transformers import AutoModelForCausalLM; "
        "m=AutoModelForCausalLM.from_pretrained(r'''%s''', trust_remote_code=True, local_files_only=True); "
        "print(type(m).__name__)"
    ) % td
    out = __import__("subprocess").run(
        [os.environ.get("PYTHON", "python"), "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "SophiaForCausalLM"
"""
    # save_pretrained writes a tqdm bar into the inherited stdout, and pytest
    # decodes that fd as UTF-8. On a zh-CN Windows host the child encodes the
    # bar's block characters as GBK, pytest's capture raises UnicodeDecodeError
    # on teardown, and every later test in the session errors in setup and
    # teardown -- 1,641 errors from one progress bar. Pin the child's IO
    # encoding so the bytes match what the reader assumes.
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=os.getcwd(),
        env={**os.environ, "PYTHON": sys.executable, "PYTHONIOENCODING": "utf-8"},
    )


def test_export_with_layers_saves_pretrained(tmp_path) -> None:
    from transformers import AutoModelForCausalLM

    from ml.integrations.export.runtime_packager import write_remote_code_bundle
    from ml.integrations.adapters.hf.loaders import ensure_auto_map
    from ml.integrations.adapters.hf.model import SophiaForCausalLM, SophiaConfig

    cfg = SophiaConfig(
        vocab_size=128,
        dim=64,
        n_layers=4,
        num_heads=2,
        head_dim=32,
        ffn_hidden=128,
        kda_decay_rank=16,
        kda_output_gate_rank=16,
        mla_q_rank=16,
        mla_kv_rank=16,
    )
    model = SophiaForCausalLM(cfg).to(device="cpu", dtype=torch.float32).eval()
    ensure_auto_map(model)

    model.save_pretrained(tmp_path, safe_serialization=True)
    write_remote_code_bundle(output_dir=str(tmp_path))

    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    assert (tmp_path / "config.json").exists()
    assert type(loaded).__name__ == "SophiaForCausalLM"


def test_prefill_runtime_cache_start_pos_zero_keep_one_uses_single_last_logit() -> None:
    import torch

    from ml.integrations.adapters.hf.loaders import prefill_runtime_cache

    class _Model:
        def __init__(self) -> None:
            self.args = type("Args", (), {"max_seq_len": 4})()
            self.calls: list[tuple[tuple[int, ...], int, bool]] = []

        def runtime_max_seq_len(self) -> int:
            return int(self.args.max_seq_len)

        def replay_with_cache(self, input_ids, start_pos=0, return_all_logits=True):
            self.calls.append(
                (
                    tuple(int(token) for token in input_ids[0].tolist()),
                    int(start_pos),
                    bool(return_all_logits),
                )
            )
            if return_all_logits:
                logits = torch.arange(
                    int(input_ids.size(1)) * 5,
                    dtype=torch.float32,
                ).reshape(1, int(input_ids.size(1)), 5)
            else:
                logits = torch.arange(5, dtype=torch.float32).reshape(1, 5)
            return logits, None

    model = _Model()
    logits = prefill_runtime_cache(
        model,
        torch.tensor([[3, 7, 11, 19, 23, 29]], dtype=torch.long),
        start_pos=0,
        logits_to_keep=1,
    )

    assert tuple(logits.shape) == (1, 1, 5)
    assert model.calls == [((3, 7, 11, 19, 23, 29), 0, False)]
