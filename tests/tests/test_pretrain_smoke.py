from __future__ import annotations

import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as functional

from ml.training.pretrain.engine.loop_execution import LoopConfig, train_loop
from ml.training.pretrain.engine.step_runner_impl import EagerStepRunner


class _ToyDataIter:
    def __init__(self) -> None:
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def state_dict(self) -> dict[str, int]:
        return {"index": int(self._index)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._index = int(state.get("index", 0) or 0)


class _ToyLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 16)
        self.output = torch.nn.Linear(16, 128, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        compute_loss: bool = False,
    ) -> SimpleNamespace:
        if not bool(compute_loss) or labels is None:
            raise ValueError("_ToyLossModel requires labels and compute_loss=True")
        logits = self.output(self.embedding(input_ids))
        loss = functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, int(logits.size(-1))),
            labels[:, 1:].reshape(-1),
        )
        return SimpleNamespace(loss=loss)


def test_pretrain_train_loop_smoke_cpu() -> None:
    torch.manual_seed(0)
    model = _ToyLossModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runner = EagerStepRunner(
        model=model,
        base_dtype=torch.float32,
        token_weighted=True,
    )
    data_iter = _ToyDataIter()

    before = model.embedding.weight.detach().clone()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LoopConfig(
            output_dir=tmp,
            max_steps=1,
            accumulation_steps=1,
            max_grad_norm=0.0,
            log_interval=0,
            save_interval=0,
            save_total_limit=2,
            tokens_per_update=8,
            token_weighted_loss=1,
        )
        result = train_loop(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            data_iter=data_iter,
            runner=runner,
            device=torch.device("cpu"),
            cfg=cfg,
            start_step=0,
            args_dict={"unit": 1},
            tokenizer=None,
            safe_serialization=False,
            export_model_artifacts_fn=lambda **_kwargs: None,
            eval_fn=None,
        )

    after = model.embedding.weight.detach()
    assert int(result.last_step) == 1
    assert int(result.train_state["seen_supervised_tokens"]) == 7
    assert not torch.equal(before, after)
