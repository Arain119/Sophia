from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from ml.tooling import cli as tooling_cli


def _pythonpath_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    package_root = str(repo_root.resolve())
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_project_script_targets_are_importable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with (repo_root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    scripts = pyproject["project"]["scripts"]

    assert "ml-pretrain-check" in scripts
    assert "ml-pretrain-validate" not in scripts

    for script_name, target in scripts.items():
        module_name, sep, function_name = str(target).partition(":")
        assert sep == ":", f"{script_name} target must be module:function"
        assert importlib.util.find_spec(module_name) is not None, script_name
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name, None)), script_name


@pytest.mark.parametrize(
    "module_name",
    [
        "ml.cli.train",
        "ml.cli.eval",
        "ml.cli.shard_builder",
        "ml.cli.pretrain_check",
        "ml.cli.sft",
        "ml.cli.rehearsal",
        "ml.tooling.cli",
    ],
)
def test_console_entrypoint_modules_support_help(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=_pythonpath_env(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "module_name",
    sorted({str(command.target).split(":", 1)[0] for command in tooling_cli._COMMANDS}),
)
def test_tooling_target_modules_support_help(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=_pythonpath_env(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_pretrain_mix_token_shards_help_exposes_source_allowlist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.tooling.pipelines.pretrain_mix_token_shards",
            "--help",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=_pythonpath_env(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--source_allowlist" in proc.stdout
    assert "--source_allowlist_fill_budget" in proc.stdout
    assert "--only_sources" not in proc.stdout
