from __future__ import annotations

from ml.core.spec import ModelSpec
from ml.tooling.scripts import enumerate_model_architectures as mod


def _tiny_spec() -> ModelSpec:
    return ModelSpec(
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
        max_seq_len=32,
    )


def test_instantiated_count_deduplicates_tied_embeddings() -> None:
    spec = _tiny_spec()

    actual = mod.count_instantiated_parameters(spec)
    embedding = int(spec.vocab_size) * int(spec.dim)
    inner = int(spec.num_heads) * int(spec.head_dim)
    ffn = 3 * int(spec.dim) * int(spec.ffn_hidden)
    kda = (
        3 * int(spec.dim) * inner
        + 3 * inner * int(spec.short_conv_kernel)
        + int(spec.dim) * int(spec.kda_decay_rank)
        + int(spec.kda_decay_rank) * inner
        + int(spec.dim) * int(spec.num_heads)
        + int(spec.num_heads)
        + inner
        + int(spec.dim) * inner
        + int(spec.head_dim)
        + inner * int(spec.dim)
        + ffn
        + 6 * int(spec.dim)
    )
    mla = (
        int(spec.dim) * int(spec.mla_q_rank)
        + int(spec.mla_q_rank)
        + int(spec.mla_q_rank) * inner
        + int(spec.dim) * int(spec.mla_kv_rank)
        + int(spec.mla_kv_rank)
        + 2 * int(spec.mla_kv_rank) * inner
        + 2 * int(spec.dim) * inner
        + ffn
        + 6 * int(spec.dim)
    )
    final_norm_and_attn_res = 3 * int(spec.dim)
    expected = embedding + 3 * kda + mla + final_norm_and_attn_res

    assert actual == expected


def test_enumeration_filters_range_and_recommends_closest_candidate() -> None:
    base = _tiny_spec()
    count_four_layers = mod.count_instantiated_parameters(base)
    count_five_layers = mod.count_instantiated_parameters(
        base.with_overrides(n_layers=5)
    )

    report = mod.enumerate_architectures(
        base_spec=base,
        vocab_sizes=[128],
        layer_counts=[4, 5],
        ffn_hidden_sizes=[128],
        target_parameters=count_five_layers,
        minimum_parameters=count_four_layers,
        maximum_parameters=count_five_layers,
    )

    assert report["schema"] == mod.REPORT_SCHEMA
    assert report["evaluated_count"] == 2
    assert report["candidate_count"] == 2
    assert report["recommended"]["parameter_count"] == count_five_layers
    assert report["recommended"]["target_delta"] == 0
    assert report["recommended_by_vocab"]["128"] == report["recommended"]


def test_main_creates_output_parent(tmp_path) -> None:
    output = tmp_path / "new" / "architecture.json"

    result = mod.main(
        [
            "--vocab-sizes",
            "128",
            "--layer-counts",
            "4",
            "--ffn-hidden-sizes",
            "128",
            "--target-parameters",
            "1000000000",
            "--minimum-parameters",
            "1",
            "--maximum-parameters",
            "1000000000",
            "--max-seq-len",
            "32",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.is_file()
