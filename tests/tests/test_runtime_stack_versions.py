from __future__ import annotations

import pathlib

from ml.runtime import stack_env, stack_versions


def test_project_python_floor_matches_numpy_pin() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.12,<3.13"' in pyproject
    assert "Python `>=3.12,<3.13`" in readme
    assert stack_env.require_python.__kwdefaults__ == {"min_version": (3, 12)}


def test_standard_stack_requires_python_312(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_require_python(*, min_version=(0, 0)) -> None:
        seen["min_version"] = tuple(min_version)

    monkeypatch.setattr(stack_versions, "require_python", _fake_require_python)
    monkeypatch.setattr(stack_versions, "require_linux", lambda *, for_training=True: None)
    monkeypatch.setattr(stack_versions, "require_numpy_supported", lambda *, max_major=2: None)
    monkeypatch.setattr(
        stack_versions,
        "require_package_exact_version",
        lambda **kwargs: "ok",
    )
    monkeypatch.setattr(
        stack_versions,
        "require_package_exact_version",
        lambda **kwargs: "ok",
    )

    stack_versions.ensure_standard_stack(mode="tools", require_tensorboard=False)

    assert seen["min_version"] == (3, 12)


def test_windows_tools_stack_does_not_require_linux_only_liger(monkeypatch) -> None:
    requested: list[str] = []

    monkeypatch.setattr(stack_versions.sys, "platform", "win32")
    monkeypatch.setattr(stack_versions, "require_python", lambda **_kwargs: None)
    monkeypatch.setattr(stack_versions, "require_numpy_supported", lambda **_kwargs: None)
    monkeypatch.setattr(
        stack_versions,
        "require_package_exact_version",
        lambda **kwargs: requested.append(str(kwargs["package"])) or "ok",
    )

    versions = stack_versions.ensure_standard_stack(mode="tools")

    assert "liger-kernel" not in requested
    assert "liger-kernel" not in versions


def test_training_stack_always_requires_liger(monkeypatch) -> None:
    requested: list[str] = []

    monkeypatch.setattr(stack_versions.sys, "platform", "win32")
    monkeypatch.setattr(stack_versions, "require_linux", lambda **_kwargs: None)
    monkeypatch.setattr(stack_versions, "require_python", lambda **_kwargs: None)
    monkeypatch.setattr(stack_versions, "require_numpy_supported", lambda **_kwargs: None)
    monkeypatch.setattr(
        stack_versions,
        "require_package_exact_version",
        lambda **kwargs: requested.append(str(kwargs["package"])) or "ok",
    )

    stack_versions.ensure_standard_stack(mode="train")

    assert "liger-kernel" in requested


def test_low_cpu_noise_env_rejects_invalid_existing_value(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "0")

    try:
        stack_env.set_low_cpu_noise_env()
    except RuntimeError as exc:
        assert str(exc) == "OMP_NUM_THREADS must be >= 1, got 0"
    else:
        raise AssertionError("invalid OMP_NUM_THREADS must fail explicitly")
