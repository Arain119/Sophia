from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_BASH = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"


def test_one_click_requires_pretraining_release_evidence() -> None:
    script = (REPO_ROOT / "tools/one_click_train.sh").read_text(encoding="utf-8")

    assert "SOPHIA_MACHINE_RECIPE is required" in script
    assert "SOPHIA_MUON_PROBE_AUDIT" in script
    assert "SOPHIA_BASE_GATE_REPORT" not in script
    assert "SOPHIA_DATA_PATH is required" in script
    assert "overwrite_output_dir" not in script
    assert "--preflight" in script
    assert "--stage" not in script
    assert "--init" not in script
    assert "--muon_probe_audit_json" in script
    assert "sophia-one-click-preflight." in script
    assert "--preflight 1" in script


def test_target_measurement_is_fixed_to_native_4k() -> None:
    preparation = (REPO_ROOT / "tools/prepare_pretrain_target_recipe.sh").read_text(
        encoding="utf-8"
    )
    assert "SOPHIA_MACHINE_SWEEP" not in preparation
    assert "--batch_size" not in preparation
    assert "--accumulation_steps" not in preparation
    assert "--step_execution_backend" not in preparation


@pytest.mark.parametrize("argument", ["unsupported", "release", "evaluate"])
def test_one_click_rejects_positional_arguments(argument: str) -> None:
    if not GIT_BASH.is_file():
        pytest.skip("Git Bash is required to execute shell entrypoint tests")
    proc = subprocess.run(
        [str(GIT_BASH), "tools/one_click_train.sh", argument, "--dry-run"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert f"unknown argument: {argument}" in proc.stderr
