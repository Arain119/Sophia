from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from ml.tooling import cli as mod


def test_tooling_cli_dispatches_registered_command(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["posttrain", "profile", "--dataset_root", "dataset"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.data.production.profile_posttrain_lengths:main"
    assert seen["argv"] == ["--dataset_root", "dataset"]
    assert seen["prog"] == "ml-tool posttrain profile"


def test_tooling_cli_dispatches_write_length_curriculum(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["posttrain", "curriculum", "--output", "dataset/posttrain_length_curriculum.json"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.data.production.write_posttrain_length_curriculum:main"
    assert seen["argv"] == ["--output", "dataset/posttrain_length_curriculum.json"]
    assert seen["prog"] == "ml-tool posttrain curriculum"


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


def test_tooling_cli_dispatches_pretrain_step_breakdown(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["pretrain", "breakdown", "--cases", "torch:torch"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.pretrain_step_breakdown:main"
    assert seen["argv"] == ["--cases", "torch:torch"]
    assert seen["prog"] == "ml-tool pretrain breakdown"


def test_tooling_cli_dispatches_pretrain_gpu_sweep(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["pretrain", "gpu-sweep", "--batch_sizes", "4"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.pretrain_gpu_sweep:main"
    assert seen["argv"] == ["--batch_sizes", "4"]
    assert seen["prog"] == "ml-tool pretrain gpu-sweep"


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


def test_tooling_cli_dispatches_sft_machine_selection(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["posttrain", "select", "--device", "cuda:0"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.machine_recipes.select_sft_machine_recipe:main"
    assert seen["argv"] == ["--device", "cuda:0"]
    assert seen["prog"] == "ml-tool posttrain select"


def test_tooling_cli_dispatches_export_posttrain_machine_recipe(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["posttrain", "export", "--stage", "sft"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.machine_recipes.export_machine_recipe:main"
    assert seen["argv"] == ["--stage", "sft"]
    assert seen["prog"] == "ml-tool posttrain export"


def test_tooling_cli_dispatches_training_release_audit(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["audit", "release", "--output_json", "out/audit.json"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.release_gate.audit_training_release:main"
    assert seen["argv"] == ["--output_json", "out/audit.json"]
    assert seen["prog"] == "ml-tool audit release"


def test_tooling_cli_dispatches_resume_drill_audit(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(target: str, argv: list[str], *, prog: str) -> int:
        seen["target"] = target
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 0

    monkeypatch.setattr(mod, "_run_target", _fake_run)

    code = mod.main(["audit", "resume", "--stage", "sft"])

    assert code == 0
    assert seen["target"] == "ml.tooling.scripts.release_gate.audit_resume_drill:main"
    assert seen["argv"] == ["--stage", "sft"]
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
