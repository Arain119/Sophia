import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.engine.checkpoints import (
    _optimizer_diagnostics,
    maybe_run_extra_evals,
)
from ml.training.pretrain.engine.io import (
    _AsyncCheckpointWriter,
    _MetricsLogger,
    _TensorBoardLogger,
    _copy_best_checkpoint,
    _read_best_metric_value,
)
from ml.training.pretrain.engine import io as checkpoint_io_mod
from ml.training.pretrain.engine import loop_execution as loop_execution_mod
from ml.training.pretrain.engine.loop_execution import LoopConfig, train_loop
from ml.training.pretrain.loop_runtime import (
    build_pretrain_eval_fn,
    build_pretrain_test_eval_fn,
)
from ml.training.pretrain.engine.eval import evaluate_loss
from ml.training.pretrain.engine.step_runner_impl import EagerStepRunner, StepRunner


class _ToyDataIter:
    def __init__(self) -> None:
        self._i = 0
        self.history: list[int] = []

    def __iter__(self):
        return self

    def __next__(self):
        self._i += 1
        self.history.append(int(self._i))
        return {
            "input_ids": torch.ones((2, 1), dtype=torch.float32),
            "labels": torch.ones((2, 1), dtype=torch.float32),
        }

    def state_dict(self) -> dict[str, int]:
        return {"index": int(self._i)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._i = int(state.get("index", 0) or 0)


class _CountingToyDataIter(_ToyDataIter):
    def __init__(self) -> None:
        super().__init__()
        self.state_dict_calls = 0

    def state_dict(self) -> dict[str, int]:
        self.state_dict_calls += 1
        return super().state_dict()


class _ToyRunner(StepRunner):
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    @property
    def zero_grad_set_to_none(self) -> bool:
        return True

    def run_micro(self, batch: dict[str, torch.Tensor], *, accum_steps: int) -> torch.Tensor:
        pred = self._model(batch["input_ids"])
        loss = (pred - batch["labels"]).pow(2).mean()
        (loss / float(max(int(accum_steps), 1))).backward()
        return loss.detach()


class _FixedGradRunner(StepRunner):
    def __init__(self, model: torch.nn.Module, *, loss: float, grad: float) -> None:
        self._model = model
        self._loss = float(loss)
        self._grad = float(grad)

    @property
    def zero_grad_set_to_none(self) -> bool:
        return True

    def run_micro(self, batch: dict[str, torch.Tensor], *, accum_steps: int) -> torch.Tensor:
        del batch, accum_steps
        for parameter in self._model.parameters():
            parameter.grad = torch.full_like(parameter, self._grad)
        first_parameter = next(self._model.parameters())
        return first_parameter.new_tensor(self._loss)


class _NonFiniteUpdateOptimizer(torch.optim.SGD):
    def step(self, closure=None):  # type: ignore[override]
        result = super().step(closure)
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.data.fill_(float("nan"))
        return result


class _ExportableToyModel(torch.nn.Linear):
    def save_pretrained(self, *_args, **_kwargs) -> None:
        return None


class _ExportableTokenizer:
    def save_pretrained(self, *_args, **_kwargs) -> None:
        return None


class _NoSnapshotLaggedRouteModel(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1, bias=False)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("rollback snapshot should not be taken")


def test_maybe_run_extra_evals_logs_due_metric() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = _MetricsLogger(
            output_dir=tmp,
            args_dict={"unit": 1},
            start_step=0,
            max_steps=4,
            device=torch.device("cpu"),
            async_write=False,
        )
        tb = _TensorBoardLogger(
            output_dir=tmp,
            args_dict={"unit": 1},
            start_step=0,
        )

        maybe_run_extra_evals(
            global_step=2,
            max_steps=4,
            extra_evals=[("test_loss", 2, lambda: 1.25)],
            metrics=metrics,
            tb=tb,
        )
        metrics.close()
        tb.close()

        with open(os.path.join(tmp, "metrics.jsonl"), encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        scalars = [event for event in events if str(event.get("type")) == "metric"]
        assert scalars[-1]["name"] == "test_loss"
        assert float(scalars[-1]["value"]) == pytest.approx(1.25)


def test_maybe_run_extra_evals_raises_on_eval_failure() -> None:
    def fail_eval() -> float:
        raise ValueError("broken eval")

    with tempfile.TemporaryDirectory() as tmp:
        metrics = _MetricsLogger(
            output_dir=tmp,
            args_dict={"unit": 1},
            start_step=0,
            max_steps=4,
            device=torch.device("cpu"),
            async_write=False,
        )
        tb = _TensorBoardLogger(
            output_dir=tmp,
            args_dict={"unit": 1},
            start_step=0,
        )
        try:
            with pytest.raises(RuntimeError, match="extra eval failed"):
                maybe_run_extra_evals(
                    global_step=2,
                    max_steps=4,
                    extra_evals=[("test_loss", 2, fail_eval)],
                    metrics=metrics,
                    tb=tb,
                )
        finally:
            metrics.close()
            tb.close()


def test_train_loop_saves_best_checkpoint_and_metric() -> None:
    model = _ExportableToyModel(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = _ToyRunner(model)
    data_iter = _ToyDataIter()
    eval_values = iter([3.0, 2.0, 2.5])

    def eval_fn() -> float:
        return float(next(eval_values))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LoopConfig(
            output_dir=tmp,
            max_steps=3,
            accumulation_steps=1,
            log_interval=0,
            save_interval=0,
            save_total_limit=5,
            tokens_per_update=1,
            eval_interval=1,
            save_best=1,
        )
        train_loop(
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
            eval_fn=eval_fn,
        )

        best_metric_path = os.path.join(tmp, "best_metric.json")
        assert os.path.exists(best_metric_path)
        with open(best_metric_path, encoding="utf-8") as f:
            obj = json.load(f)
        assert int(obj["step"]) == 2
        assert abs(float(obj["val_loss"]) - 2.0) < 1e-6

        best_ckpt = os.path.join(tmp, "checkpoints", "best.pt")
        final_ckpt = os.path.join(tmp, "checkpoints", "ckpt_step3.pt")
        assert os.path.exists(best_ckpt)
        assert os.path.exists(f"{best_ckpt}.sha256")
        with open(f"{best_ckpt}.sha256", encoding="ascii") as handle:
            assert handle.read().rstrip().endswith("  best.pt")
        assert os.path.exists(final_ckpt)


def test_train_loop_resume_keeps_prior_best_metric() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = _ToyRunner(model)
    data_iter = _ToyDataIter()
    eval_values = iter([2.5])

    def eval_fn() -> float:
        return float(next(eval_values))

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "best_metric.json"), "w", encoding="utf-8") as f:
            json.dump({"step": 2, "val_loss": 2.0}, f)
        cfg = LoopConfig(
            output_dir=tmp,
            max_steps=3,
            accumulation_steps=1,
            log_interval=0,
            save_interval=0,
            save_total_limit=5,
            tokens_per_update=1,
            eval_interval=1,
            save_best=1,
        )
        train_loop(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            data_iter=data_iter,
            runner=runner,
            device=torch.device("cpu"),
            cfg=cfg,
            start_step=2,
            args_dict={"unit": 1},
            tokenizer=None,
            safe_serialization=False,
            export_model_artifacts_fn=lambda **_kwargs: None,
            eval_fn=eval_fn,
            resume_train_state={"best_eval_loss": 2.0, "seen_supervised_tokens": 0},
        )

        with open(os.path.join(tmp, "best_metric.json"), encoding="utf-8") as f:
            obj = json.load(f)
        assert int(obj["step"]) == 2
        assert abs(float(obj["val_loss"]) - 2.0) < 1e-6


def test_best_metric_reader_rejects_corrupt_json(tmp_path) -> None:
    (tmp_path / "best_metric.json").write_text("{", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to read best metric file"):
        _read_best_metric_value(output_dir=str(tmp_path), metric_name="val_loss")


def test_copy_best_checkpoint_requires_saved_source(tmp_path) -> None:
    (tmp_path / "checkpoints").mkdir()

    with pytest.raises(FileNotFoundError, match="best checkpoint source"):
        _copy_best_checkpoint(output_dir=str(tmp_path), step=7)


def test_optimizer_diagnostics_failure_is_not_silenced() -> None:
    class _BrokenDiagnostics:
        def diagnostics(self):
            raise RuntimeError("broken")

    with pytest.raises(RuntimeError, match="optimizer diagnostics failed"):
        _optimizer_diagnostics(_BrokenDiagnostics())


def test_metrics_logger_preserves_initial_run_args_and_meta_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger_a = _MetricsLogger(
            output_dir=tmp,
            args_dict={
                "stage": "first",
                "batch_size": 8,
                "seq_len": 1024,
                "accumulation_steps": 4,
                "target_tokens_per_update": 32768,
                "data_path": "/tmp/train_manifest.json",
                "tokenizer_path": "/tmp/tokenizer",
                "_sophia_train_manifest_sha1": "sha1-a",
                "_sophia_run_kind": "pretrain",
                "_sophia_base_dtype": "bf16",
                "_sophia_precision_stack": "amp_bf16",
                "_sophia_machine_signature": "host-a|cpu",
                "_sophia_resolved_resume_checkpoint": "",
            },
            start_step=0,
            max_steps=10,
            device=torch.device("cpu"),
            async_write=False,
        )
        logger_a.close()

        logger_b = _MetricsLogger(
            output_dir=tmp,
            args_dict={
                "stage": "second",
                "batch_size": 2,
                "seq_len": 2048,
                "accumulation_steps": 16,
                "target_tokens_per_update": 65536,
                "data_path": "/tmp/train_manifest.json",
                "tokenizer_path": "/tmp/tokenizer",
                "_sophia_train_manifest_sha1": "sha1-a",
                "_sophia_run_kind": "pretrain",
                "_sophia_base_dtype": "bf16",
                "_sophia_precision_stack": "amp_bf16",
                "_sophia_machine_signature": "host-b|cuda:0",
                "_sophia_resolved_resume_checkpoint": "/tmp/ckpt_step7.pt",
            },
            start_step=7,
            max_steps=10,
            device=torch.device("cpu"),
            async_write=False,
        )
        logger_b.close()

        with open(os.path.join(tmp, "run_args.json"), encoding="utf-8") as handle:
            run_args = json.load(handle)
        with open(os.path.join(tmp, "run_meta.json"), encoding="utf-8") as handle:
            run_meta = json.load(handle)

        assert str(run_args["stage"]) == "first"
        assert int(run_args["batch_size"]) == 8
        assert int(run_meta["start_step"]) == 0
        assert int(run_meta["max_steps"]) == 10
        assert str(run_meta["_sophia_machine_signature"]) == "host-a|cpu"
        assert str(run_meta["_sophia_resolved_resume_checkpoint"]) == ""
        assert int(run_meta["recipe"]["seq_len"]) == 1024
        assert int(run_meta["recipe"]["batch_size"]) == 8
        assert str(run_meta["input_provenance"]["_sophia_train_manifest_sha1"]) == "sha1-a"

        with open(os.path.join(tmp, "metrics.jsonl"), encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        start_events = [event for event in events if str(event.get("type")) == "start"]
        assert len(start_events) == 2
        assert int(start_events[0]["start_step"]) == 0
        assert str(start_events[0]["_sophia_machine_signature"]) == "host-a|cpu"
        assert str(start_events[0]["_sophia_resolved_resume_checkpoint"]) == ""
        assert int(start_events[1]["start_step"]) == 7
        assert str(start_events[1]["_sophia_machine_signature"]) == "host-b|cuda:0"
        assert str(start_events[1]["_sophia_resolved_resume_checkpoint"]) == "/tmp/ckpt_step7.pt"
        assert str(start_events[1]["_sophia_run_kind"]) == "pretrain"
        assert str(start_events[1]["_sophia_base_dtype"]) == "bf16"
        assert str(start_events[1]["_sophia_precision_stack"]) == "amp_bf16"
        assert int(start_events[1]["recipe"]["seq_len"]) == 2048
        assert int(start_events[1]["recipe"]["batch_size"]) == 2
        assert int(start_events[1]["recipe"]["accumulation_steps"]) == 16
        assert int(start_events[1]["recipe"]["target_tokens_per_update"]) == 65536
        assert "index_topk" not in start_events[1]["recipe"]
        assert str(start_events[1]["input_provenance"]["data_path"]) == "/tmp/train_manifest.json"
        assert str(start_events[1]["input_provenance"]["tokenizer_path"]) == "/tmp/tokenizer"
        assert (
            str(start_events[1]["input_provenance"]["_sophia_train_manifest_sha1"])
            == "sha1-a"
        )


def test_train_loop_saves_final_checkpoint_before_model_export() -> None:
    model = _ExportableToyModel(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = _ToyRunner(model)
    data_iter = _ToyDataIter()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LoopConfig(
            output_dir=tmp,
            max_steps=2,
            accumulation_steps=1,
            log_interval=0,
            save_interval=0,
            save_total_limit=5,
            tokens_per_update=1,
            enable_checkpoints=1,
        )
        try:
            train_loop(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                data_iter=data_iter,
                runner=runner,
                device=torch.device("cpu"),
                cfg=cfg,
                start_step=0,
                args_dict={"unit": 1},
                tokenizer=_ExportableTokenizer(),
                safe_serialization=False,
                export_model_artifacts_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("export failed")),
                eval_fn=None,
            )
        except RuntimeError as exc:
            assert "export failed" in str(exc)
        else:
            raise AssertionError("expected export failure")

        final_ckpt = os.path.join(tmp, "checkpoints", "ckpt_step2.pt")
        assert os.path.exists(final_ckpt)


def test_async_checkpoint_writer_propagates_background_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fail_save(**_kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_io_mod, "save_checkpoint_state", fail_save)
    writer = _AsyncCheckpointWriter()
    writer.save(
        output_dir=str(tmp_path),
        step=4,
        state={},
        save_total_limit=1,
    )
    assert writer._thread is not None
    writer._thread.join(timeout=5.0)

    with pytest.raises(RuntimeError, match="Async checkpoint save failed"):
        writer.raise_if_failed()


def test_async_checkpoint_failure_stops_before_next_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FailingWriter:
        def __init__(self) -> None:
            self.checks = 0

        def raise_if_failed(self) -> None:
            self.checks += 1
            if self.checks == 2:
                raise RuntimeError("Async checkpoint save failed")

        def save(self, **_kwargs) -> None:
            return None

        def wait(self) -> None:
            return None

    monkeypatch.setattr(loop_execution_mod, "_AsyncCheckpointWriter", _FailingWriter)
    model = _ExportableToyModel(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data_iter = _ToyDataIter()

    with pytest.raises(RuntimeError, match="Async checkpoint save failed"):
        train_loop(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            data_iter=data_iter,
            runner=_ToyRunner(model),
            device=torch.device("cpu"),
            cfg=LoopConfig(
                output_dir=str(tmp_path),
                max_steps=2,
                accumulation_steps=1,
                log_interval=0,
                save_interval=1,
                save_total_limit=1,
                tokens_per_update=1,
                enable_checkpoints=1,
                async_checkpoint=1,
            ),
            start_step=0,
            args_dict={"unit": 1},
            tokenizer=None,
            safe_serialization=False,
            export_model_artifacts_fn=lambda **_kwargs: None,
        )

    assert data_iter.history == [1]


def test_final_checkpoint_failure_prevents_model_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fail_save(**_kwargs) -> None:
        raise OSError("final checkpoint failed")

    monkeypatch.setattr(checkpoint_io_mod, "save_checkpoint_state", fail_save)
    model = _ExportableToyModel(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    exported: list[bool] = []

    with pytest.raises(RuntimeError, match="Async checkpoint save failed"):
        train_loop(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            data_iter=_ToyDataIter(),
            runner=_ToyRunner(model),
            device=torch.device("cpu"),
            cfg=LoopConfig(
                output_dir=str(tmp_path),
                max_steps=1,
                accumulation_steps=1,
                log_interval=0,
                save_interval=0,
                save_total_limit=1,
                tokens_per_update=1,
                enable_checkpoints=1,
                async_checkpoint=1,
            ),
            start_step=0,
            args_dict={"unit": 1},
            tokenizer=_ExportableTokenizer(),
            safe_serialization=False,
            export_model_artifacts_fn=lambda **_kwargs: exported.append(True),
        )

    assert exported == []


def test_train_loop_defers_data_iter_snapshot_until_needed() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = _ToyRunner(model)
    data_iter = _CountingToyDataIter()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LoopConfig(
            output_dir=tmp,
            max_steps=3,
            accumulation_steps=1,
            log_interval=0,
            save_interval=0,
            save_total_limit=1,
            tokens_per_update=1,
            enable_checkpoints=0,
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

    assert data_iter.state_dict_calls == 2
    assert result.train_state["data_iter_state"] == {"index": 3}
@pytest.mark.parametrize(
    ("loss", "grad", "reason"),
    [
        (float("nan"), 1.0, "nonfinite_loss"),
        (0.25, float("inf"), "nonfinite_grad_norm"),
    ],
)
def test_train_loop_finite_guard_stops_before_optimizer_step(
    loss: float,
    grad: float,
    reason: str,
) -> None:
    model = _ExportableToyModel(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = _FixedGradRunner(model, loss=loss, grad=grad)
    data_iter = _ToyDataIter()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LoopConfig(
            output_dir=tmp,
            max_steps=1,
            accumulation_steps=1,
            log_interval=0,
            save_interval=0,
            save_total_limit=1,
            tokens_per_update=1,
            enable_checkpoints=0,
        )
        with pytest.raises(RuntimeError, match="stability guard tripped"):
            train_loop(
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

        incident_path = os.path.join(tmp, "stability_incident.json")
        assert os.path.exists(incident_path)
        with open(incident_path, encoding="utf-8") as f:
            incident = json.load(f)
        assert incident["reason"] == reason
        assert int(incident["step"]) == 1
        assert float(model.weight.detach().item()) == pytest.approx(3.0)


def test_train_loop_finite_guard_stops_after_nonfinite_update(
    tmp_path,
) -> None:
    model = _ExportableToyModel(1, 1, bias=False)
    optimizer = _NonFiniteUpdateOptimizer(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="nonfinite_parameter_or_optimizer_state"):
        train_loop(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            data_iter=_ToyDataIter(),
            runner=_FixedGradRunner(model, loss=0.25, grad=1.0),
            device=torch.device("cpu"),
            cfg=LoopConfig(
                output_dir=str(tmp_path),
                max_steps=1,
                accumulation_steps=1,
                log_interval=0,
                save_interval=0,
                save_total_limit=1,
                tokens_per_update=1,
                enable_checkpoints=0,
            ),
            start_step=0,
            args_dict={"unit": 1},
            tokenizer=None,
            safe_serialization=False,
            export_model_artifacts_fn=lambda **_kwargs: None,
        )

    incident = json.loads(
        (tmp_path / "stability_incident.json").read_text(encoding="utf-8")
    )
    assert incident["reason"] == "nonfinite_parameter_or_optimizer_state"
    assert int(incident["step"]) == 1


def test_evaluate_loss_requests_compute_loss() -> None:
    seen: list[bool] = []

    class _ComputeLossModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor, compute_loss: bool = False):
            del input_ids, labels
            seen.append(bool(compute_loss))
            return SimpleNamespace(loss=self.weight.new_tensor(0.5))

    model = _ComputeLossModel().eval()
    data_iter = iter(
        [
            {
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "labels": torch.ones((1, 4), dtype=torch.long),
            }
        ]
    )

    loss = evaluate_loss(
        model=model,
        data_iter=data_iter,
        steps=1,
        base_dtype=torch.float32,
    )

    assert loss == 0.5
    assert seen == [True]


def test_evaluate_loss_rejects_models_without_compute_loss() -> None:
    class _ComputeLossIncompatibleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor):
            del input_ids, labels
            return SimpleNamespace(loss=self.weight.new_tensor(0.25))

    model = _ComputeLossIncompatibleModel().eval()
    data_iter = iter(
        [
            {
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "labels": torch.ones((1, 4), dtype=torch.long),
            }
        ]
    )

    try:
        _ = evaluate_loss(
            model=model,
            data_iter=data_iter,
            steps=1,
            base_dtype=torch.float32,
        )
    except TypeError as exc:
        assert "compute_loss=True" in str(exc)
        return
    raise AssertionError("expected evaluate_loss to reject models without compute_loss")


def test_eager_step_runner_requests_compute_loss() -> None:
    seen: list[bool] = []

    class _ComputeLossModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor, compute_loss: bool = False):
            del input_ids, labels
            seen.append(bool(compute_loss))
            return SimpleNamespace(loss=self.weight + 1.0)

    model = _ComputeLossModel().train(True)
    runner = EagerStepRunner(
        model=model,
        base_dtype=torch.float32,
        token_weighted=False,
    )
    batch = {
        "input_ids": torch.ones((1, 4), dtype=torch.int32),
        "labels": torch.ones((1, 4), dtype=torch.int32),
    }
    loss = runner.run_micro(batch, accum_steps=1)
    assert torch.is_tensor(loss)
    assert seen == [True]


def test_eager_step_runner_rejects_models_without_compute_loss() -> None:
    class _IncompatibleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor):
            del input_ids, labels
            return SimpleNamespace(loss=self.weight + 1.0)

    with pytest.raises(TypeError, match="compute_loss=True"):
        EagerStepRunner(
            model=_IncompatibleModel().train(True),
            base_dtype=torch.float32,
            token_weighted=False,
        )


def test_build_pretrain_eval_fn_uses_current_evaluate_loss_interface(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dtype": "int32",
                "tokenizer_sha1": "0" * 40,
                "shards": [{"path": "shard_00000.bin", "tokens": 16}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "shard_00000.bin").write_bytes(b"\x00" * 32)

    args = PretrainRunConfig(
        data_path="dataset/pretrain_tokens",
        eval_interval=1,
        eval_steps=2,
        eval_data_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        batch_size=1,
        seed=0,
    )
    model = torch.nn.Linear(1, 1, bias=False)
    seen: dict[str, object] = {}

    class _ClosableIter:
        def __iter__(self):
            return self

        def __next__(self):
            return {
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "labels": torch.ones((1, 4), dtype=torch.long),
            }

        def close(self) -> None:
            seen["closed"] = True

    with patch(
        "ml.training.pretrain.loop_runtime.validate_tokenizer_fingerprint",
        lambda **_kwargs: None,
    ), patch(
        "ml.training.pretrain.loop_runtime.build_pretrain_data_iter",
        lambda **_kwargs: _ClosableIter(),
    ), patch(
        "ml.training.pretrain.loop_runtime.evaluate_loss",
        lambda **kwargs: seen.update(kwargs) or 0.25,
    ):
        eval_fn, interval = build_pretrain_eval_fn(
            args=args,
            seq_len=4,
            model=model,
            device=torch.device("cpu"),
            base_dtype=torch.float32,
        )

        assert interval == 1
        assert eval_fn is not None
        assert float(eval_fn()) == 0.25

    assert "data_iter" in seen
    assert seen["steps"] == 2
    assert seen["base_dtype"] == torch.float32
    assert seen.get("closed") is True


def test_build_pretrain_test_eval_fn_uses_current_evaluate_loss_interface(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dtype": "int32",
                "tokenizer_sha1": "0" * 40,
                "shards": [{"path": "shard_00000.bin", "tokens": 16}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "shard_00000.bin").write_bytes(b"\x00" * 32)

    args = PretrainRunConfig(
        data_path="dataset/pretrain_tokens",
        test_data_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        batch_size=1,
        seed=0,
    )
    model = torch.nn.Linear(1, 1, bias=False)
    seen: dict[str, object] = {}

    class _ClosableIter:
        def __iter__(self):
            return self

        def __next__(self):
            return {
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "labels": torch.ones((1, 4), dtype=torch.long),
            }

        def close(self) -> None:
            seen["closed"] = True

    with patch(
        "ml.training.pretrain.loop_runtime.validate_tokenizer_fingerprint",
        lambda **_kwargs: None,
    ), patch(
        "ml.training.pretrain.loop_runtime.build_pretrain_data_iter",
        lambda **_kwargs: _ClosableIter(),
    ), patch(
        "ml.training.pretrain.loop_runtime.evaluate_loss",
        lambda **kwargs: seen.update(kwargs) or 0.5,
    ):
        eval_fn = build_pretrain_test_eval_fn(
            args=args,
            seq_len=4,
            model=model,
            device=torch.device("cpu"),
            base_dtype=torch.float32,
            steps=3,
        )

        assert eval_fn is not None
        assert float(eval_fn()) == 0.5

    assert "data_iter" in seen
    assert seen["steps"] == 3
    assert seen["base_dtype"] == torch.float32
    assert seen.get("closed") is True


def test_build_pretrain_test_eval_fn_returns_none_when_steps_disabled(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dtype": "int32",
                "tokenizer_sha1": "0" * 40,
                "shards": [{"path": "shard_00000.bin", "tokens": 16}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "shard_00000.bin").write_bytes(b"\x00" * 32)

    args = PretrainRunConfig(
        data_path="dataset/pretrain_tokens",
        test_data_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        batch_size=1,
        seed=0,
    )
    model = torch.nn.Linear(1, 1, bias=False)

    eval_fn = build_pretrain_test_eval_fn(
        args=args,
        seq_len=4,
        model=model,
        device=torch.device("cpu"),
        base_dtype=torch.float32,
        steps=0,
    )

    assert eval_fn is None
