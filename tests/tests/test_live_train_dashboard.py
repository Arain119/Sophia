import json
from pathlib import Path

from tools.live_train_dashboard import (
    _checkpoint_age_s,
    _checkpoint_write_active,
    _errors,
    _is_train_command,
    _load_metrics,
    _resolved_max_steps,
    _tokens_per_update,
)


def test_train_command_recognizes_all_supported_training_entrypoints() -> None:
    for module in ("ml.cli.train",):
        assert _is_train_command(["python", "-m", module, "--output_dir", "/tmp/run"])


def test_train_command_rejects_non_training_processes() -> None:
    assert not _is_train_command(["python", "tools/train_monitor_loop.py"])
    assert not _is_train_command(["python", "-m", "ml.tooling.cli"])


def test_tokens_per_update_reads_signed_recipe_target() -> None:
    assert _tokens_per_update({"recipe": {"target_tokens_per_update": 32_768}}) == 32_768


def test_tokens_per_update_prefers_runtime_target() -> None:
    assert (
        _tokens_per_update(
            {
                "target_tokens_per_update": 16_384,
                "recipe": {"target_tokens_per_update": 32_768},
            }
        )
        == 16_384
    )


def test_resolved_max_steps_prefers_effective_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "effective_config.json").write_text(
        json.dumps({"max_steps": 100}), encoding="utf-8"
    )
    assert _resolved_max_steps(run_dir, 60) == 100


def test_load_metrics_keeps_latest_branch_for_each_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = (
        {"type": "train", "step": 10, "loss": 3.0},
        {"type": "train", "step": 20, "loss": 8.0},
        {"type": "eval", "step": 20, "val_loss": 4.0},
        {"type": "start", "start_step": 10, "max_steps": 100},
        {"type": "train", "step": 20, "loss": 2.5},
        {"type": "eval", "step": 20, "val_loss": 3.5},
    )
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    latest_start, latest_train, history, eval_history = _load_metrics(run_dir)

    assert latest_start["start_step"] == 10
    assert latest_train["loss"] == 2.5
    assert [(row["step"], row["loss"]) for row in history] == [
        (10, 3.0),
        (20, 2.5),
    ]
    assert [(row["step"], row["val_loss"]) for row in eval_history] == [(20, 3.5)]


def test_load_metrics_keeps_early_training_after_log_exceeds_tail_limit(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = run_dir / "metrics.jsonl"
    rows = [{"type": "train", "step": 10, "loss": 4.0}]
    rows.extend({"type": "metric", "step": i} for i in range(50_001))
    rows.append({"type": "train", "step": 20, "loss": 3.0})
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    _start, _latest, history, _eval_history = _load_metrics(run_dir)

    assert [row["step"] for row in history] == [10, 20]


def test_errors_ignore_failures_before_current_launch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    launch_log = tmp_path / "run.launch.log"
    launch_log.write_text(
        "[PREFLIGHT] old run\n"
        "Traceback: old failure\n"
        "[PREFLIGHT] current run\n"
        "[TRAIN] step=10 loss=3.0\n",
        encoding="utf-8",
    )

    assert _errors(run_dir, launch_log) == []


def test_errors_report_failures_in_current_launch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    launch_log = tmp_path / "run.launch.log"
    launch_log.write_text(
        "[PREFLIGHT] old run\n"
        "Traceback: old failure\n"
        "[PREFLIGHT] current run\n"
        "RuntimeError: current failure\n",
        encoding="utf-8",
    )

    assert _errors(run_dir, launch_log) == [
        "run.launch.log: RuntimeError: current failure"
    ]


def test_checkpoint_age_starts_at_current_run(monkeypatch) -> None:
    monkeypatch.setattr("tools.live_train_dashboard.time.time", lambda: 1_500.0)

    age = _checkpoint_age_s([{"mtime": 100.0}], {"time": 1_400.0})

    assert age == 100.0


def test_checkpoint_write_active_detects_recent_atomic_temp_file(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    temp = checkpoint_dir / "ckpt_step600.pt.tmp.123.abc"
    temp.write_bytes(b"partial")
    monkeypatch.setattr("tools.live_train_dashboard.time.time", lambda: temp.stat().st_mtime)

    assert _checkpoint_write_active(tmp_path)


def test_checkpoint_write_active_ignores_completed_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "ckpt_step600.pt").write_bytes(b"complete")

    assert not _checkpoint_write_active(tmp_path)
