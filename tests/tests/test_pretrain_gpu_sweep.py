from __future__ import annotations

import argparse

import pytest

from ml.tooling.scripts import pretrain_gpu_sweep as mod


def test_gpu_sweep_builds_target_token_candidates() -> None:
    candidates = mod.build_candidates(
        batch_sizes=(4, 5),
        loss_chunks=(0, 512),
        muon_ortho_batch_maxes=(1, 8),
        seq_len=4096,
        target_tokens_per_update=270336,
    )

    names = [candidate.name for candidate in candidates]

    assert "bs4_acc17_chunk512_muonmb1" in names
    assert "bs5_acc14_chunk512_muonmb8" in names


def test_parse_int_csv_rejects_empty_values() -> None:
    try:
        mod._parse_int_csv("")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_gpu_sweep_ranks_by_throughput_then_memory(monkeypatch) -> None:
    def fake_benchmark(case, **kwargs):
        batch_size = int(kwargs["batch_size"])
        return {
            "ok": True,
            "candidate_name": f"bs{batch_size}",
            "batch_size": batch_size,
            "tokens_per_s": 100.0 if batch_size in {2, 3} else 90.0,
            "peak_gb": 50.0 if batch_size == 3 else 40.0,
            "loss_chunk_size": int(case.loss_chunk_size),
        }

    monkeypatch.setattr(mod, "_device_total_memory_gb", lambda _device: 80.0)
    monkeypatch.setattr(mod, "benchmark_case", fake_benchmark)

    payload = mod.run_sweep(
        argparse.Namespace(
            data_path="dataset/pretrain_tokens",
            seq_len=4096,
            target_tokens_per_update=270336,
            batch_sizes="2,3,4",
            loss_chunks="0",
            muon_ortho_batch_maxes="1",
            warmup_updates=0,
            timed_updates=1,
            include_optimizer=1,
            device="cuda:0",
            seed=42,
            step_execution_backend="inductor",
        )
    )

    assert payload["best"]["batch_size"] == 3
    assert payload["memory_best"]["batch_size"] == 3
    assert payload["best"]["memory_utilization"] == pytest.approx(0.625)


def test_gpu_sweep_records_oom_and_raises_other_failures(monkeypatch) -> None:
    def fake_benchmark(case, **kwargs):
        del case
        batch_size = int(kwargs["batch_size"])
        if batch_size == 3:
            raise RuntimeError("CUDA out of memory")
        if batch_size == 4:
            raise RuntimeError("shape mismatch")
        return {
            "ok": True,
            "candidate_name": f"bs{batch_size}",
            "batch_size": batch_size,
            "tokens_per_s": 100.0,
            "peak_gb": 40.0,
            "peak_reserved_gb": 44.0,
        }

    monkeypatch.setattr(mod, "_device_total_memory_gb", lambda _device: 80.0)
    monkeypatch.setattr(mod, "benchmark_case", fake_benchmark)
    args = argparse.Namespace(
        data_path="dataset/pretrain_tokens",
        seq_len=4096,
        target_tokens_per_update=270336,
        batch_sizes="2,3",
        loss_chunks="0",
        muon_ortho_batch_maxes="1",
        warmup_updates=0,
        timed_updates=1,
        include_optimizer=1,
        device="cuda:0",
        seed=42,
        step_execution_backend="inductor",
    )

    payload = mod.run_sweep(args)

    assert [row["ok"] for row in payload["results"][:2]] == [True, False]
    assert "out of memory" in str(payload["results"][1]["error"]).lower()
    assert [row["phase"] for row in payload["results"][2:]] == [
        "compile_refine",
        "compile_refine",
    ]
    assert payload["best"]["allocated_memory_utilization"] == pytest.approx(0.5)
    assert payload["best"]["reserved_memory_utilization"] == pytest.approx(0.55)

    args.batch_sizes = "4"
    with pytest.raises(RuntimeError, match="shape mismatch"):
        mod.run_sweep(args)
