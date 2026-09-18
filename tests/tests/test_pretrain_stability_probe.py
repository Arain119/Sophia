from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.tooling.scripts import pretrain_stability_probe as mod
from ml.tooling.scripts.pretrain_stability_probe import (
    build_parser,
    run_probe,
    summarize_metrics,
)


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_summarize_metrics_passes_stable_run(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics,
        [
            {"type": "start", "max_steps": 3},
            {
                "type": "train",
                "step": 1,
                "max_steps": 3,
                "loss": 11.0,
                "grad_norm": 6.0,
                "tok_s": 100.0,
            },
            {
                "type": "train",
                "step": 2,
                "max_steps": 3,
                "loss": 10.8,
                "grad_norm": 5.0,
                "tok_s": 110.0,
            },
            {
                "type": "train",
                "step": 3,
                "max_steps": 3,
                "loss": 10.7,
                "grad_norm": 4.0,
                "tok_s": 120.0,
            },
            {"type": "end", "step": 3, "max_steps": 3},
        ],
    )

    summary = summarize_metrics(metrics)

    assert summary["passed"] is True
    assert summary["max_loss"] == 11.0
    assert summary["observed_max_grad_norm"] == 6.0


def test_summarize_metrics_accepts_large_finite_values(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics,
        [
            {"type": "start", "max_steps": 4},
            {
                "type": "train",
                "step": 1,
                "max_steps": 4,
                "loss": 11.0,
                "grad_norm": 21.0,
            },
            {
                "type": "train",
                "step": 2,
                "max_steps": 4,
                "loss": 13.5,
                "grad_norm": 22.0,
            },
            {
                "type": "train",
                "step": 3,
                "max_steps": 4,
                "loss": 12.0,
                "grad_norm": 23.0,
            },
            {
                "type": "train",
                "step": 4,
                "max_steps": 4,
                "loss": 12.0,
                "grad_norm": 51.0,
            },
            {"type": "end", "step": 4, "max_steps": 4},
        ],
    )

    summary = summarize_metrics(metrics)

    assert summary["passed"] is True
    assert summary["nonfinite_loss_steps"] == []
    assert summary["nonfinite_grad_norm_steps"] == []


def test_summarize_metrics_rejects_nonfinite_values(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics,
        [
            {"type": "start", "max_steps": 2},
            {"type": "train", "step": 1, "loss": float("nan"), "grad_norm": 1.0},
            {"type": "train", "step": 2, "loss": 1.0, "grad_norm": float("inf")},
            {"type": "end", "step": 2, "max_steps": 2},
        ],
    )

    summary = summarize_metrics(metrics)

    assert summary["passed"] is False
    assert summary["nonfinite_loss_steps"] == [1]
    assert summary["nonfinite_grad_norm_steps"] == [2]


def test_summarize_metrics_tracks_complete_attention_telemetry(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    attention_rows = [
        {
            "type": "metric",
            "step": 1,
            "name": f"attention/mla_layer_{layer}/head_{head}_max_logit",
            "value": float(layer + head),
        }
        for layer in range(7)
        for head in range(16)
    ]
    _write_metrics(
        metrics,
        [
            {"type": "start", "max_steps": 1},
            {
                "type": "train",
                "step": 1,
                "max_steps": 1,
                "loss": 10.0,
                "grad_norm": 2.0,
            },
            *attention_rows,
            {"type": "end", "step": 1, "max_steps": 1},
        ],
    )

    summary = summarize_metrics(metrics)

    assert summary["attention_logit_metric_count"] == 112
    assert summary["attention_logit_telemetry_complete"] is True
    assert summary["nonfinite_attention_logit_steps"] == []


def test_stability_release_binding_hashes_protocol_and_checkpoint(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    checkpoint = tmp_path / "checkpoint.bin"
    protocol.write_bytes(b"protocol")
    checkpoint.write_bytes(b"checkpoint")
    binding = mod._evaluation_binding(
        protocol_json=protocol,
        checkpoint=checkpoint,
        checkpoint_sha256=None,
        eval_export_model=None,
    )
    assert binding["protocol_path"] == str(protocol.resolve())
    assert binding["protocol_sha256"] == hashlib.sha256(b"protocol").hexdigest()
    assert binding["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert binding["eval_export_model_path"] is None
    with pytest.raises(ValueError, match="requires"):
        mod._evaluation_binding(
            protocol_json=protocol,
            checkpoint=None,
            checkpoint_sha256=None,
            eval_export_model=None,
        )
    with pytest.raises(ValueError, match="not applicable"):
        mod._evaluation_binding(
            protocol_json=protocol,
            checkpoint=checkpoint,
            checkpoint_sha256=None,
            eval_export_model=tmp_path / "model.bin",
        )


def test_formal_probe_parser_uses_derived_warmup_boundary() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args(["--machine_recipe_json", "recipe.json"])

    assert parsed.max_steps == 154
    assert parsed.checkpoint_interval == 153
    assert parsed.overwrite_output_dir == 0


def test_formal_probe_rejects_short_or_destructive_run(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    parser = build_parser()
    short = parser.parse_args(
        ["--machine_recipe_json", str(recipe), "--max_steps", "153"]
    )
    with pytest.raises(ValueError, match="post-warmup"):
        run_probe(short)

    destructive = parser.parse_args(
        ["--machine_recipe_json", str(recipe), "--overwrite_output_dir", "1"]
    )
    with pytest.raises(ValueError, match="forbidden"):
        run_probe(destructive)


def test_formal_probe_uses_full_20b_lr_schedule_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "probe"
    captured: dict[str, int] = {}

    def fake_build_pretrain_args(**kwargs):
        return SimpleNamespace(
            output_dir=kwargs["output_dir"],
            total_tokens=20_000_000_000,
            target_tokens_per_update=1_310_720,
            warmup_steps=153,
        )

    def fake_run(config) -> dict[str, int]:
        captured["horizon"] = int(config._sophia_lr_schedule_horizon_steps)
        captured["total_tokens"] = int(config.total_tokens)
        captured["max_steps"] = int(config.max_steps)
        output = Path(config.output_dir)
        output.mkdir(parents=True)
        _write_metrics(
            output / "metrics.jsonl",
            [
                {"type": "start", "max_steps": 154},
                {
                    "type": "train",
                    "step": 1,
                    "max_steps": 154,
                    "loss": 10.0,
                    "grad_norm": 2.0,
                    "tok_s": 100.0,
                },
                {
                    "type": "train",
                    "step": 154,
                    "max_steps": 154,
                    "loss": 8.0,
                    "grad_norm": 2.0,
                    "tok_s": 100.0,
                },
                {"type": "end", "step": 154, "max_steps": 154},
            ],
        )
        return {
            "recapture_peak_allocated_bytes": 123,
            "recapture_peak_reserved_bytes": 456,
        }

    monkeypatch.setattr(mod, "build_pretrain_args", fake_build_pretrain_args)
    monkeypatch.setattr(mod, "run", fake_run)
    args = build_parser().parse_args(
        [
            "--machine_recipe_json",
            str(recipe),
            "--output_dir",
            str(output_dir),
            "--max_steps",
            "154",
        ]
    )

    result = run_probe(args)

    assert result["passed"] is True
    assert captured == {
        "horizon": 15_259,
        "total_tokens": 201_850_880,
        "max_steps": 154,
    }


def test_graph_recapture_probe_requires_and_reports_capture_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "recapture"

    def fake_build_pretrain_args(**kwargs):
        return SimpleNamespace(
            output_dir=kwargs["output_dir"],
            total_tokens=20_000_000_000,
            target_tokens_per_update=1_310_720,
            warmup_steps=153,
        )

    def fake_run(config) -> dict[str, int]:
        output = Path(config.output_dir)
        output.mkdir(parents=True)
        _write_metrics(
            output / "metrics.jsonl",
            [
                {"type": "start", "max_steps": 12},
                {
                    "type": "train",
                    "step": 10,
                    "max_steps": 12,
                    "loss": 10.0,
                    "grad_norm": 2.0,
                    "tok_s": 100.0,
                },
                {"type": "eval", "step": 10, "val_loss": 9.0},
                {
                    "type": "train",
                    "step": 11,
                    "max_steps": 12,
                    "loss": 9.0,
                    "grad_norm": 2.0,
                    "tok_s": 100.0,
                },
                {"type": "end", "step": 12, "max_steps": 12},
            ],
        )
        return {
            "recapture_peak_allocated_bytes": 123,
            "recapture_peak_reserved_bytes": 456,
        }

    monkeypatch.setattr(mod, "build_pretrain_args", fake_build_pretrain_args)
    monkeypatch.setattr(mod, "run", fake_run)
    args = build_parser().parse_args(
        [
            "--machine_recipe_json",
            str(recipe),
            "--output_dir",
            str(output_dir),
            "--graph_recapture_probe",
            "1",
            "--max_steps",
            "12",
            "--eval_interval",
            "10",
        ]
    )

    result = run_probe(args)

    assert result["passed"] is True
    assert result["graph_recapture_covered"] is True
    assert result["graph_recapture_peak_allocated_bytes"] == 123
    assert result["graph_recapture_peak_reserved_bytes"] == 456
