from __future__ import annotations

import numpy as np
import pytest

from ml.tooling.scripts.data import build_tokenizer_shard_heldout_suite as mod


def test_source_shards_reconstructs_exact_source_boundaries(tmp_path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(np.asarray([1, 2, 3], dtype="<i4").tobytes())
    second.write_bytes(np.asarray([4, 5], dtype="<i4").tobytes())
    manifest = {
        "shards": [
            {"path": first.name, "tokens": 3},
            {"path": second.name, "tokens": 2},
        ]
    }
    report = {
        "shards": [
            {"path": first.name, "tokens": 3, "sha256": "ignored"},
            {"path": second.name, "tokens": 2, "sha256": "ignored"},
        ],
        "tokens_by": {"source": {"source_a": 3, "source_b": 2}},
    }

    source_shards, source_tokens = mod._source_shards(
        shard_root=tmp_path,
        manifest=manifest,
        build_report=report,
    )

    assert source_tokens == {"source_a": 3, "source_b": 2}
    assert source_shards["source_a"] == [(first, 3)]
    assert source_shards["source_b"] == [(second, 2)]


def test_source_domain_config_must_be_an_exact_partition() -> None:
    payload = {
        "domains": {
            "native_chinese": ["zh"],
            "english": ["en"],
            "code": ["code"],
            "math": ["math"],
        }
    }
    domains = mod._validate_domains(
        source_domains=payload,
        available_sources={"zh", "en", "code", "math"},
    )
    assert domains["native_chinese"] == ["zh"]

    payload["domains"]["math"] = ["code"]
    with pytest.raises(ValueError, match="multiple domains"):
        mod._validate_domains(
            source_domains=payload,
            available_sources={"zh", "en", "code", "math"},
        )


def test_document_sampling_respects_eos_boundaries(tmp_path) -> None:
    shard = tmp_path / "shard.bin"
    shard.write_bytes(
        np.asarray([10, 11, 3, 20, 21, 22, 3], dtype="<i4").tobytes()
    )

    sampled = mod._sample_document(
        source="source",
        sample_index=0,
        shards=[(shard, 7)],
        eos_token_id=3,
        max_scan_tokens=16,
    )

    assert sampled in ([10, 11], [20, 21, 22])
