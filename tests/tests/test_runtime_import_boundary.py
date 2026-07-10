"""
Import-boundary guard for the self-contained runtime.

The active runtime implementation (``ml/runtime/model/transformer.py``) is copied flat
into exported HF model dirs as ``sophia_runtime.py`` by
``ml.integrations.export.runtime_packager``. For that export to stay
self-contained — and to keep the runtime cheap to import — its import closure
must not pull in:

- ``ml.training`` (never bundled into exported dirs), or
- ``transformers`` / the HF model wrapper (the runtime depends only on
  ``rope`` / ``runtime_linear`` and runtime helpers).

These run in a fresh interpreter so an unrelated earlier import in the test
process cannot mask a real regression.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _sophia_modules_after_importing(import_stmt: str) -> set[str]:
    code = textwrap.dedent(
        f"""
        import sys
        {import_stmt}
        for name in sorted(sys.modules):
            print(name)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


def test_runtime_import_does_not_pull_training_or_transformers() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in this environment")

    loaded = _sophia_modules_after_importing(
        "import ml.runtime.model.transformer"
    )

    leaked_training = sorted(m for m in loaded if m.startswith("ml.training"))
    assert not leaked_training, (
        "runtime import pulled in the training package (breaks self-contained "
        f"export): {leaked_training}"
    )

    assert "transformers" not in loaded, (
        "runtime import pulled in transformers — the runtime must stay importable "
        "without the HF stack"
    )
    assert "ml.integrations.adapters.hf.model" not in loaded, (
        "runtime import pulled in the HF model wrapper"
    )


def test_pretrain_pipeline_does_not_eagerly_import_hf_wrapper_at_module_top_level() -> None:
    pipeline_path = (
        Path(__file__).resolve().parents[2]
        / "ml"
        / "tasks"
        / "pretrain"
        / "pipeline.py"
    )
    source = pipeline_path.read_text(encoding="utf-8")

    assert "from ml.integrations.adapters.hf.model import" not in source


def test_task_and_inference_paths_do_not_source_model_loading_from_hf_adapter() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "ml" / "runtime" / "inference" / "eval" / "model_io.py",
        repo_root / "ml" / "training" / "posttrain" / "runtime.py",
    ]
    forbidden = "load_trainable_decoder"
    forbidden_import = "from ml.integrations.adapters.hf.loaders import"

    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert f"{forbidden_import} {forbidden}" not in source, str(path)


def test_task_and_inference_paths_do_not_source_local_tokenizer_loading_from_hf_adapter() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "ml" / "runtime" / "inference" / "eval" / "common.py",
        repo_root / "ml" / "runtime" / "inference" / "eval" / "model_io.py",
        repo_root / "ml" / "training" / "posttrain" / "runtime.py",
    ]
    forbidden_import = "from ml.integrations.adapters.hf.loaders import load_tokenizer"

    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert forbidden_import not in source, str(path)


def test_export_model_dir_does_not_depend_on_hf_support_wrapper() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "ml"
        / "integrations"
        / "export"
        / "model_dir.py"
    )
    source = path.read_text(encoding="utf-8")

    assert "from ml.integrations.adapters.hf.config import" not in source


def test_training_runtime_exports_do_not_depend_on_hf_export_adapter() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "ml" / "training" / "posttrain" / "runtime.py",
        repo_root / "ml" / "training" / "pretrain" / "loop_runtime.py",
    ]
    forbidden_import = "from ml.integrations.adapters.hf.export import"

    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert forbidden_import not in source, str(path)


def test_hf_wrapper_implementation_does_not_depend_on_export_or_training_layers() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "ml"
        / "integrations"
        / "adapters"
        / "hf"
        / "model.py"
    )
    source = path.read_text(encoding="utf-8")

    assert "from ml.integrations.export" not in source
    assert "from ml.training" not in source
    assert "from ml.tasks" not in source
