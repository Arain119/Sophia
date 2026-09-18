from __future__ import annotations

import torch

from ml.training.pretrain.engine.sm120_graph_runner import SM120GraphRunner


class _Graph:
    def __init__(self) -> None:
        self.replays = 0
        self.resets = 0

    def replay(self) -> None:
        self.replays += 1

    def reset(self) -> None:
        self.resets += 1


def test_sm120_graph_composes_formal_update_from_ten_graph_blocks() -> None:
    runner = SM120GraphRunner.__new__(SM120GraphRunner)
    runner._device = torch.device("cpu")
    runner._graph = None
    runner._graph_ready = False
    runner._static_results = ()
    runner._attention_logit_max = torch.empty((0, 0))
    graph = _Graph()
    calls = {"fills": 0, "captures": 0}

    def fill_slab(_data_iter) -> None:
        calls["fills"] += 1

    def capture_graph() -> None:
        calls["captures"] += 1
        runner._graph = graph
        runner._graph_ready = True
        runner._static_results = (
            (torch.tensor(2.0), torch.tensor(3, dtype=torch.int64)),
        )

    runner._fill_slab = fill_slab
    runner._capture_graph = capture_graph

    result = runner.run_update(
        data_iter=iter(()),
        accumulation_steps=320,
        device=torch.device("cpu"),
        token_weighted_cfg=True,
    )

    assert calls == {"fills": 10, "captures": 1}
    assert graph.replays == 9
    assert result.token_weighted_active
    assert float(result.update_loss_sum) == 20.0
    assert int(result.update_supervised_tokens) == 30


def test_sm120_graph_rejects_old_64_micro_update() -> None:
    runner = SM120GraphRunner.__new__(SM120GraphRunner)

    try:
        runner.begin_update(accum_steps=64)
    except RuntimeError as exc:
        assert "accumulation_steps=320" in str(exc)
    else:
        raise AssertionError("old accumulation contract was accepted")


def test_sm120_graph_materializes_fresh_optimizer_state() -> None:
    runner = SM120GraphRunner.__new__(SM120GraphRunner)
    runner._optimizer = None
    calls: list[str] = []

    class _Optimizer:
        def initialize_state(self) -> None:
            calls.append("initialize")

    optimizer = _Optimizer()
    runner.set_optimizer(optimizer)  # type: ignore[arg-type]

    assert runner._optimizer is optimizer
    assert calls == ["initialize"]


def test_sm120_graph_arms_allocator_release_for_initial_capture(
    monkeypatch,
) -> None:
    runner = SM120GraphRunner.__new__(SM120GraphRunner)
    runner._has_captured_graph = False
    runner._optimizer = None
    runner._optimizer_state_offloaded = False
    runner._static_batches = ({},)
    output = torch.nn.Linear(1, 1)
    runner._model = type(
        "_Model",
        (),
        {"get_output_embeddings": lambda _self: output},
    )()
    armed: list[bool] = []

    def stop_after_first_micro(_batch) -> None:
        armed.append(
            bool(
                getattr(
                    output,
                    "_sophia_release_cuda_cache_before_linear_ce",
                    False,
                )
            )
        )
        raise RuntimeError("stop after first warmup micro")

    runner._run_static_micro = stop_after_first_micro

    try:
        runner._capture_graph()
    except RuntimeError as exc:
        assert str(exc) == "stop after first warmup micro"
    else:
        raise AssertionError("capture warmup did not run")

    assert armed == [True]


def test_sm120_graph_arms_allocator_release_before_recapture_warmup(
    monkeypatch,
) -> None:
    runner = SM120GraphRunner.__new__(SM120GraphRunner)
    runner._device = torch.device("cpu")
    runner._has_captured_graph = True
    runner._optimizer = None
    runner._optimizer_state_offloaded = False
    runner._static_batches = ({},)
    offloads: list[str] = []
    runner._offload_optimizer_state_for_capture = lambda: offloads.append("offload")
    peak_resets: list[torch.device] = []
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: peak_resets.append(device),
    )
    output = torch.nn.Linear(1, 1)
    runner._model = type(
        "_Model",
        (),
        {"get_output_embeddings": lambda _self: output},
    )()
    armed: list[bool] = []

    def stop_after_first_micro(_batch) -> None:
        armed.append(
            bool(
                getattr(
                    output,
                    "_sophia_release_cuda_cache_before_linear_ce",
                    False,
                )
            )
        )
        raise RuntimeError("stop after first warmup micro")

    runner._run_static_micro = stop_after_first_micro

    try:
        runner._capture_graph()
    except RuntimeError as exc:
        assert str(exc) == "stop after first warmup micro"
    else:
        raise AssertionError("capture warmup did not run")

    assert armed == [True]
    assert offloads == ["offload"]
    assert peak_resets == [torch.device("cpu")]


def test_sm120_graph_releases_capture_before_evaluation(
    monkeypatch,
) -> None:
    runner = SM120GraphRunner.__new__(SM120GraphRunner)
    graph = _Graph()
    model = torch.nn.Linear(2, 2)
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    main_grad = torch.full_like(parameter, 3.0, dtype=torch.float32)
    runner._device = torch.device("cpu")
    runner._model = model
    runner._main_grads = (main_grad,)
    runner._graph = graph
    runner._graph_ready = True
    runner._static_results = ((torch.tensor(1.0), torch.tensor(1)),)
    runner._optimizer = None
    runner._optimizer_state_offloaded = False
    calls: list[str] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: calls.append(f"synchronize:{device}"),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    runner.prepare_for_evaluation()

    assert graph.resets == 1
    assert runner._graph is None
    assert runner._graph_ready is False
    assert runner._static_results == ()
    assert parameter.grad is None
    assert runner._main_grads[0] is main_grad
    assert torch.equal(main_grad, torch.full_like(main_grad, 3.0))
    assert calls == ["synchronize:cpu", "empty"]


def test_sm120_graph_recaptures_after_evaluation(monkeypatch) -> None:
    runner = SM120GraphRunner.__new__(SM120GraphRunner)
    old_graph = _Graph()
    new_graph = _Graph()
    model = torch.nn.Linear(1, 1, bias=False)
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    runner._device = torch.device("cpu")
    runner._model = model
    runner._main_grads = (torch.zeros_like(parameter, dtype=torch.float32),)
    runner._graph = old_graph
    runner._graph_ready = True
    runner._static_results = ((torch.tensor(1.0), torch.tensor(1)),)
    runner._optimizer = None
    runner._optimizer_state_offloaded = False
    runner._attention_logit_max = torch.empty((0, 0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    runner.prepare_for_evaluation()

    calls = {"captures": 0, "fills": 0}

    def fill_slab(_data_iter) -> None:
        calls["fills"] += 1

    def capture_graph() -> None:
        calls["captures"] += 1
        runner._graph = new_graph
        runner._graph_ready = True
        runner._static_results = (
            (torch.tensor(2.0), torch.tensor(3, dtype=torch.int64)),
        )

    runner._fill_slab = fill_slab
    runner._capture_graph = capture_graph

    runner.run_update(
        data_iter=iter(()),
        accumulation_steps=320,
        device=torch.device("cpu"),
        token_weighted_cfg=True,
    )

    assert old_graph.resets == 1
    assert old_graph.replays == 0
    assert runner._graph is new_graph
    assert new_graph.replays == 9
    assert calls == {"captures": 1, "fills": 10}
