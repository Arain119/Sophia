from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from ml.training.rehearsal.defaults import REPO_ROOT
from ml.training.rehearsal.io import append_jsonl
from ml.training.rehearsal.plan import RehearsalStage


def terminate_process(
    proc: subprocess.Popen[Any],
    *,
    interrupt: bool,
) -> int | None:
    try:
        if bool(interrupt):
            proc.send_signal(signal.SIGINT)
        else:
            proc.terminate()
    except Exception:
        pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            return int(rc)
        time.sleep(0.1)
    try:
        proc.kill()
    except Exception:
        pass
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            return int(rc)
        time.sleep(0.1)
    return None


def run_stage_process(
    *,
    stage: RehearsalStage,
    stdout_handle: Any,
    stderr_handle: Any,
    progress_interval_seconds: float,
    events_path: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> int:
    env = dict(os.environ)
    repo_src = str(REPO_ROOT)
    current_pythonpath = str(env.get("PYTHONPATH", "") or "")
    parts = [part for part in current_pythonpath.split(os.pathsep) if part]
    parts = [part for part in parts if os.path.abspath(part) != os.path.abspath(repo_src)]
    env["PYTHONPATH"] = os.pathsep.join([repo_src, *parts])
    proc = subprocess.Popen(
        list(stage.argv),
        cwd=str(REPO_ROOT),
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    started = time.monotonic()
    next_progress = (
        started + float(progress_interval_seconds)
        if float(progress_interval_seconds) > 0.0
        else None
    )
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return int(rc)
            now = time.monotonic()
            if next_progress is not None and now >= next_progress:
                elapsed_s = int(max(now - started, 0.0))
                append_jsonl(
                    events_path,
                    {
                        "stage": stage.name,
                        "status": "running",
                        "time": float(time.time()),
                        "elapsed_s": int(elapsed_s),
                        "stdout_log": str(stdout_log),
                        "stderr_log": str(stderr_log),
                    },
                )
                print(
                    "[REHEARSAL] "
                    f"{stage.name} still running ({elapsed_s}s) | "
                    f"stdout={stdout_log} stderr={stderr_log}",
                    flush=True,
                )
                next_progress = now + float(progress_interval_seconds)
            time.sleep(1.0)
    except KeyboardInterrupt:
        terminate_process(proc, interrupt=True)
        raise


__all__ = ["run_stage_process", "subprocess", "terminate_process", "time"]
