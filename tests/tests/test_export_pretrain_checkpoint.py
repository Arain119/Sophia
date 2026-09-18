from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ml.tooling.scripts import export_pretrain_checkpoint as mod


def _prepare_export(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights_step200.pt"
    checkpoint.write_bytes(b"checkpoint")
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}", encoding="utf-8")
    (parent / "model.safetensors").write_bytes(b"parent")
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        mod,
        "load_checkpoint",
        lambda _path: SimpleNamespace(
            step=200,
            model=model.state_dict(),
            kind="full",
            args={"_sophia_machine_recipe_sha256": "f" * 64},
        ),
    )
    monkeypatch.setattr(
        mod,
        "load_trainable_decoder_from_export_dir",
        lambda **_kwargs: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(mod, "load_local_export_tokenizer", lambda _path: object())

    def fake_export(**kwargs) -> None:
        output = tmp_path / "export"
        assert str(output) == kwargs["output_dir"]
        (output / "model.safetensors").write_bytes(b"export")

    monkeypatch.setattr(mod, "export_model_artifacts", fake_export)
    return checkpoint, parent


def test_export_checkpoint_records_bound_provenance(monkeypatch, tmp_path) -> None:
    checkpoint, parent = _prepare_export(monkeypatch, tmp_path)

    report = mod.export_checkpoint(
        checkpoint_path=str(checkpoint),
        parent_export=str(parent),
        output_dir=str(tmp_path / "export"),
    )

    assert report["schema"] == mod.EXPORT_SCHEMA
    assert report["checkpoint"]["step"] == 200
    assert report["checkpoint"]["kind"] == "full"
    assert report["checkpoint"]["machine_recipe_sha256"] == "f" * 64
    assert report["model"]["sha256"] == mod.sha256_file(
        tmp_path / "export" / "model.safetensors"
    )
    assert (tmp_path / "export" / "checkpoint_export.json").is_file()


def test_export_checkpoint_rejects_nonempty_output(monkeypatch, tmp_path) -> None:
    checkpoint, parent = _prepare_export(monkeypatch, tmp_path)
    output = tmp_path / "export"
    output.mkdir()
    (output / "existing").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        mod.export_checkpoint(
            checkpoint_path=str(checkpoint),
            parent_export=str(parent),
            output_dir=str(output),
        )
