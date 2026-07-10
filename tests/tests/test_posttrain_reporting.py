from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from ml.tasks.sft.reports import write_supervised_capability_report
from ml.tasks.sft.runtime_reports import build_sft_checkpoint_report_callback
from ml.training.posttrain.data import ChatExample


class _FakeSupervisedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, compute_loss: bool = False, **_kwargs):
        assert bool(compute_loss) is True
        loss = input_ids[:, 0].float().mean()
        return SimpleNamespace(loss=loss, logits=None)


def test_write_supervised_capability_report_groups_rows(monkeypatch, tmp_path) -> None:
    examples = [
        ChatExample(
            messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            metadata={"capability_family": "translation", "bucket": "bucket_a"},
        ),
        ChatExample(
            messages=[{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
            metadata={"capability_family": "humanistic", "bucket": "bucket_b"},
        ),
    ]

    monkeypatch.setattr(
        "ml.tasks.sft.reports.load_supervised_examples",
        lambda _path: list(examples),
    )
    monkeypatch.setattr(
        "ml.tasks.sft.reports.build_single_example_supervised_batch",
        lambda **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        },
    )

    out = write_supervised_capability_report(
        output_dir=str(tmp_path),
        report_name="sft_capability_report.json",
        model=_FakeSupervisedModel(),
        tokenizer=object(),
        split_paths={"val": "dummy.jsonl"},
        max_seq_len=128,
        pad_to_multiple_of=8,
    )

    assert float(out["splits"]["val"]["overall"]["loss_mean"]) == 1.0
    assert json.loads((tmp_path / "sft_capability_report.json").read_text(encoding="utf-8"))["kind"] == "sft_capability_report"


def test_build_sft_checkpoint_report_callback_writes_step_scoped_report(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "ml.tasks.sft.runtime_reports.write_sft_report",
        lambda **kwargs: seen.update(kwargs),
    )

    callback = build_sft_checkpoint_report_callback(
        runtime=SimpleNamespace(output_dir=str(tmp_path / "release_sft")),
        model=_FakeSupervisedModel(),
        tokenizer=object(),
        config=SimpleNamespace(
            eval_data_path="eval.jsonl",
            test_data_path="test.jsonl",
        ),
        global_step=12,
    )
    callback()

    assert str(seen["output_dir"]).endswith("checkpoint_capability/step_00000012")
    assert seen["split_paths"] == {"val": "eval.jsonl", "test": "test.jsonl"}
