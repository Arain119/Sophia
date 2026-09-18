from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ml.tooling import cli as mod


def test_tooling_cli_dispatches_pretrain_machine_validation(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["pretrain", "check", "--device", "cuda:0"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.machine_recipes.check_pretrain_machine_recipe:main"
    assert seen["argv"] == ["--device", "cuda:0"]
    assert seen["prog"] == "ml-tool pretrain check"


def test_tooling_cli_dispatches_pretrain_final_check(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["pretrain", "final-check", "--evidence", "evidence.json"])

    assert code == 0
    assert seen["target"] == (
        "ml.tooling.scripts.eval_pretrain_base_capability_gate:main"
    )
    assert seen["argv"] == ["--evidence", "evidence.json"]
    assert seen["prog"] == "ml-tool pretrain final-check"


def test_tooling_cli_registers_v6_acquisition_commands() -> None:
    registered = {
        (command.domain, command.action): command.target
        for command in mod._COMMANDS
    }

    assert registered[("pretrain", "source-download")] == (
        "ml.tooling.scripts.data.download_hf_source_inventory:main"
    )
    assert registered[("pretrain", "acquisition-extend")] == (
        "ml.tooling.scripts.data.extend_pretrain_acquisition_bundle:main"
    )
    assert registered[("pretrain", "humanities-filter")] == (
        "ml.tooling.scripts.data.filter_chinese_humanities_derivative:main"
    )
    assert registered[("pretrain", "mix-materialize")] == (
        "ml.tooling.scripts.data.materialize_pretrain_mix_policy:main"
    )
    assert ("pretrain", "context-audit") not in registered
    assert ("pretrain", "architecture-sweep") not in registered


def test_tooling_cli_dispatches_pretrain_stability_probe(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["pretrain", "stability", "--max_steps", "3"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.pretrain_stability_probe:main"
    assert seen["argv"] == ["--max_steps", "3"]
    assert seen["prog"] == "ml-tool pretrain stability"


def test_tooling_cli_dispatches_resume_drill_audit(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["audit", "resume", "--stage", "pretrain"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.release_gate.audit_resume_drill:main"
    assert seen["argv"] == ["--stage", "pretrain"]
    assert seen["prog"] == "ml-tool audit resume"


def test_tooling_cli_dispatches_soak_test_audit(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["audit", "soak", "--output_json", "out/soak.json"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.release_gate.audit_soak_test:main"
    assert seen["argv"] == ["--output_json", "out/soak.json"]
    assert seen["prog"] == "ml-tool audit soak"


def test_tooling_cli_returns_usage_error_without_command() -> None:
    code = mod.main([])
    assert code == 2


def test_tooling_cli_rejects_unregistered_domain() -> None:
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["unsupported", "command"])
    assert exc_info.value.code == 2


def test_tooling_cli_module_entrypoint_has_no_runpy_warning() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    package_root = str(repo_root.resolve())
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-m", "ml.tooling.cli", "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert proc.returncode == 0
    assert "RuntimeWarning" not in proc.stderr
