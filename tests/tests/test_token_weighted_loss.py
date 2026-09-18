import tempfile
import torch

from ml.training.pretrain.engine.loop_execution import LoopConfig, train_loop
from ml.training.pretrain.engine.step_runner_impl import (
    MicroStepResult,
    StepRunner,
)


class _ToyDataIter:
    def __init__(self) -> None:
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._i += 1
        return {}

    def state_dict(self) -> dict[str, int]:
        return {"index": int(self._i)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._i = int(state.get("index", 0) or 0)


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self) -> torch.Tensor:
        return self.w


class _TokenWeightedRunner(StepRunner):
    def __init__(
        self,
        model: _ScalarModel,
        *,
        coeffs: list[float],
        supervised_tokens: list[int],
    ) -> None:
        self._model = model
        self._coeffs = list(coeffs)
        self._supervised_tokens = list(supervised_tokens)
        self._i = 0

    @property
    def zero_grad_set_to_none(self) -> bool:
        return True

    def run_micro(
        self, batch: dict[str, torch.Tensor], *, accum_steps: int
    ) -> MicroStepResult:
        idx = self._i
        self._i += 1
        coeff = float(self._coeffs[idx])
        sup = int(self._supervised_tokens[idx])

        loss = self._model.w * torch.tensor(coeff, dtype=self._model.w.dtype)
        supervised = torch.tensor(sup, dtype=torch.int64, device=loss.device)
        loss_sum = loss * supervised.to(dtype=loss.dtype)
        loss_sum.backward()
        return MicroStepResult(
            loss_detached=loss.detach(),
            loss_sum_detached=loss_sum.detach(),
            supervised_tokens=supervised.detach(),
        )


def test_token_weighted_loss_normalizes_by_supervised_tokens() -> None:
    model = _ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    runner = _TokenWeightedRunner(model, coeffs=[1.0, 3.0], supervised_tokens=[1, 3])
    data_iter = _ToyDataIter()

    # grad (unscaled) = 1*1 + 3*3 = 10; tokens=4 => scaled grad=2.5
    # SGD(lr=1): w <- w - 2.5 => -2.5
    with tempfile.TemporaryDirectory() as tmp:
        cfg = LoopConfig(
            output_dir=tmp,
            max_steps=1,
            accumulation_steps=2,
            log_interval=0,
            save_interval=0,
            save_total_limit=5,
            tokens_per_update=1,
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

    assert abs(float(model.w.detach().item()) - (-2.5)) < 1e-6
    assert int(result.train_state["seen_supervised_tokens"]) == 4
