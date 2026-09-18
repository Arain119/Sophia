from __future__ import annotations

from types import SimpleNamespace

from ml.cli import train as train_cli
from ml.tasks.pretrain import pipeline as pipeline_mod


def test_preflight_stops_before_training(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    args = object()

    monkeypatch.setattr(
        train_cli,
        "validate_selected_muon_probe_audit",
        lambda **kwargs: calls.append(("muon_audit", kwargs)),
    )
    monkeypatch.setattr(
        train_cli,
        "validate_pretrain_dataset_manifest",
        lambda **kwargs: calls.append(("dataset", kwargs)),
    )
    monkeypatch.setattr(
        train_cli,
        "build_pretrain_args",
        lambda **kwargs: calls.append(("build", kwargs)) or args,
    )
    monkeypatch.setattr(
        train_cli,
        "run_preflight",
        lambda value: calls.append(("preflight", value)),
    )
    monkeypatch.setattr(
        train_cli,
        "run",
        lambda value: calls.append(("train", value)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "ml-train",
            "--data_path",
            "/fresh/train",
            "--machine_recipe_json",
            "/recipe.json",
            "--muon_probe_audit_json",
            "/audit.json",
            "--preflight",
            "1",
        ],
    )

    train_cli.main()

    assert [name for name, _ in calls] == [
        "muon_audit",
        "dataset",
        "build",
        "preflight",
    ]


def test_normal_cli_path_still_runs_training(monkeypatch) -> None:
    calls: list[str] = []
    args = object()

    monkeypatch.setattr(
        train_cli,
        "validate_selected_muon_probe_audit",
        lambda **_: calls.append("muon_audit"),
    )
    monkeypatch.setattr(
        train_cli,
        "validate_pretrain_dataset_manifest",
        lambda **_: calls.append("dataset"),
    )
    monkeypatch.setattr(
        train_cli,
        "build_pretrain_args",
        lambda **_: calls.append("build") or args,
    )
    monkeypatch.setattr(
        train_cli,
        "run_preflight",
        lambda _: calls.append("preflight"),
    )
    monkeypatch.setattr(train_cli, "run", lambda _: calls.append("train"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "ml-train",
            "--data_path",
            "/fresh/train",
            "--machine_recipe_json",
            "/recipe.json",
        ],
    )

    train_cli.main()

    assert calls == ["muon_audit", "dataset", "build", "train"]


def test_pipeline_preflight_stops_after_data_recipe_and_model(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    args = object()
    deps = object()
    pipeline = SimpleNamespace(_MACHINE_SIGNATURE_ARG_KEY="machine-key")

    monkeypatch.setattr(pipeline_mod, "PretrainPipeline", lambda args: pipeline)
    monkeypatch.setattr(pipeline_mod, "_pretrain_deps", lambda: deps)
    monkeypatch.setattr(
        pipeline_mod.pretrain_runtime,
        "setup_runtime",
        lambda value, **kwargs: calls.append(("setup", (value, kwargs))),
    )
    monkeypatch.setattr(
        pipeline_mod.pretrain_runtime,
        "run_runtime_preflight",
        lambda value, **kwargs: calls.append(("runtime", (value, kwargs))),
    )
    monkeypatch.setattr(
        pipeline_mod.pretrain_runtime,
        "prepare_data_and_model",
        lambda value, **kwargs: calls.append(("prepare", (value, kwargs))),
    )

    pipeline_mod.run_preflight(args)

    assert [name for name, _ in calls] == ["setup", "runtime", "prepare"]
    for _, (actual_pipeline, kwargs) in calls:
        assert actual_pipeline is pipeline
        assert kwargs["deps"] is deps
