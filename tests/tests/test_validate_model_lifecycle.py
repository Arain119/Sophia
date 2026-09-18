from __future__ import annotations

import pytest
from pathlib import Path

from ml.core.spec import ModelSpec
from ml.tooling.scripts import validate_model_lifecycle as mod


def test_scaled_model_spec_preserves_selected_architecture_semantics() -> None:
    selected = ModelSpec.default()
    scaled = mod.scaled_model_spec()

    assert scaled.vocab_size == selected.vocab_size
    assert scaled.n_layers == 4
    assert [scaled.to_model_args().layer_type(index) for index in range(4)] == [
        "kda",
        "kda",
        "kda",
        "mla",
    ]
    assert scaled.kda_decay_lower_bound == selected.kda_decay_lower_bound
    assert scaled.short_conv_kernel == selected.short_conv_kernel
    assert scaled.norm_eps == selected.norm_eps
    assert scaled.dropout == selected.dropout


def test_lifecycle_detects_optional_liger_backend(monkeypatch) -> None:
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: None)
    assert mod._liger_available() is False

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: object())
    assert mod._liger_available() is True


def test_lifecycle_detects_optional_fla_backend(monkeypatch) -> None:
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: None)
    assert mod._fla_available() is False

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: object())
    assert mod._fla_available() is True


def test_lifecycle_output_rejects_host_mounted_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "path_uses_host_mounted_storage", lambda _path: True)
    with pytest.raises(ValueError, match="target-native storage"):
        mod._require_native_output(tmp_path / "run")
