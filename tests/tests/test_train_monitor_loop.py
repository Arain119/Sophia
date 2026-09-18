import subprocess
import sys
from pathlib import Path

from tools.train_monitor_loop import _issues


def test_direct_script_help_works_without_pythonpath(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "tools" / "train_monitor_loop.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-dir" in result.stdout


def _running_without_metrics(start_time: float) -> dict:
    return {
        "state": "running",
        "health": {"state": "healthy", "reasons": []},
        "start": {"time": start_time},
        "train": {},
        "history": [],
        "progress": {},
        "gpu": {"util_pct": 100},
        "history_stats": {},
    }


def test_issues_allows_initial_metric_startup_grace(monkeypatch) -> None:
    monkeypatch.setattr("tools.train_monitor_loop.time.time", lambda: 1_000.0)

    assert _issues(_running_without_metrics(900.0)) == []


def test_issues_reports_missing_metrics_after_startup_grace(monkeypatch) -> None:
    monkeypatch.setattr("tools.train_monitor_loop.time.time", lambda: 1_000.0)

    assert _issues(_running_without_metrics(600.0)) == [
        "未读取到训练指标时间戳"
    ]


def test_issues_suppresses_low_gpu_during_checkpoint_write(monkeypatch) -> None:
    monkeypatch.setattr("tools.train_monitor_loop.time.time", lambda: 1_000.0)
    data = _running_without_metrics(900.0)
    data["train"] = {"time": 990.0}
    data["gpu"] = {"util_pct": 12}
    data["progress"] = {"checkpoint_age_s": 30.0}

    assert _issues(data) == []


def test_issues_suppresses_low_gpu_while_atomic_checkpoint_is_in_progress(
    monkeypatch,
) -> None:
    monkeypatch.setattr("tools.train_monitor_loop.time.time", lambda: 1_000.0)
    data = _running_without_metrics(900.0)
    data["train"] = {"time": 990.0}
    data["gpu"] = {"util_pct": 12}
    data["progress"] = {
        "checkpoint_age_s": 600.0,
        "checkpoint_write_active": True,
    }

    assert _issues(data) == []
