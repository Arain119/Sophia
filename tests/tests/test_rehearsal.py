from __future__ import annotations

import json

import pytest

import ml.training.rehearsal as rehearsal_mod
from ml.errors import SophiaUsageError
from ml.training.rehearsal import build_rehearsal_plan, run_rehearsal
from ml.training.rehearsal.config import RehearsalConfig, resolve_rehearsal_config
from ml.training.rehearsal import process as rehearsal_process
from ml.training.posttrain.curriculum import default_posttrain_curriculum_stage
from ml.cli import rehearsal as rehearsal_cli


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_hf_config(path, *, max_seq_len: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"max_seq_len": int(max_seq_len)}, ensure_ascii=False),
        encoding="utf-8",
    )


def _make_config(tmp_path) -> RehearsalConfig:
    dataset_root = tmp_path / "dataset"
    pretrain_root = dataset_root / "pretrain"
    sft_root = dataset_root / "sft"
    for path in (pretrain_root, sft_root):
        path.mkdir(parents=True, exist_ok=True)

    _write_jsonl(sft_root / "train.jsonl", [{"messages": [{"role": "user", "content": "hi"}]}])
    _write_jsonl(sft_root / "val.jsonl", [{"messages": [{"role": "user", "content": "hi"}]}])

    curriculum_path = dataset_root / "posttrain_length_curriculum.json"
    curriculum_path.write_text(
        json.dumps(
            {
                "sft_curriculum": [{"stage": "sft_core", "max_seq_len": 4096}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return RehearsalConfig(
        output_root=str(tmp_path / "out" / "local_rehearsal"),
        device="cuda:0",
        seed=1337,
        overwrite_output_dir=True,
        pretrain_data=str(pretrain_root),
        posttrain_curriculum_path=str(curriculum_path),
        pretrain_tokens=0,
        sft_train_data=str(sft_root / "train.jsonl"),
        sft_eval_data=str(sft_root / "val.jsonl"),
        sft_test_data=str(sft_root / "val.jsonl"),
        sft_max_steps=2,
        posttrain_batch_size=1,
        posttrain_eval_steps=1,
        posttrain_report_max_examples_per_split=0,
        posttrain_report_progress_every=25,
        progress_interval_seconds=0.0,
    )


def test_build_rehearsal_plan_contains_decoder_stages(tmp_path) -> None:
    plan = build_rehearsal_plan(_make_config(tmp_path))

    assert [stage.name for stage in plan.stages] == [
        "pretrain_check",
        "sft",
    ]
    assert plan.stages[0].argv[2] == "ml.cli.pretrain_check"
    assert plan.stages[1].argv[2] == "ml.cli.sft"
    assert plan.stages[1].argv[plan.stages[1].argv.index("--export_dir") + 1].replace(
        "\\", "/"
    ).endswith("/pretrain_check")
    assert "--test_data" in plan.stages[1].argv
    assert plan.stages[1].argv[plan.stages[1].argv.index("--allow_missing_machine_recipe") + 1] == "1"
    assert "--report_max_examples_per_split" in plan.stages[1].argv
    assert "--report_progress_every" in plan.stages[1].argv
    assert plan.stages[1].argv[plan.stages[1].argv.index("--curriculum_stage") + 1] == default_posttrain_curriculum_stage(stage_kind="sft")


def test_rehearsal_cli_defaults_to_token_shards_dataset_root() -> None:
    parsed = rehearsal_cli.build_parser().parse_args([])

    assert str(parsed.pretrain_data).replace("\\", "/").endswith(
        "dataset/pretrain_tokens"
    )


def test_run_rehearsal_writes_report(
    monkeypatch,
    tmp_path,
) -> None:
    plan = build_rehearsal_plan(_make_config(tmp_path))
    seen: list[tuple[str, ...]] = []

    class _FakePopen:
        def __init__(self, argv: list[str], *, cwd: str, env, stdout, stderr) -> None:
            del stdout, stderr
            seen.append(tuple(argv))
            assert cwd
            assert str(env["PYTHONPATH"]).split(":")[0]
            module_name = str(argv[2])
            if module_name == "ml.cli.pretrain_check":
                _write_hf_config(tmp_path / "out" / "local_rehearsal" / "pretrain_check", max_seq_len=4096)
            elif module_name == "ml.cli.sft":
                _write_hf_config(
                    tmp_path / "out" / "local_rehearsal" / "sft" / "export",
                    max_seq_len=4096,
                )
            self.returncode = 0

        def poll(self) -> int:
            return int(self.returncode)

        def send_signal(self, _sig: int) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(rehearsal_process.subprocess, "Popen", _FakePopen)

    report = run_rehearsal(plan)

    assert report["status"] == "completed"
    assert [stage["name"] for stage in report["stages"]] == [
        "pretrain_check",
        "sft",
    ]
    assert [stage["status"] for stage in report["stages"]] == [
        "completed",
        "completed",
    ]
    assert len(seen) == 2
    assert seen[0][2] == "ml.cli.pretrain_check"
    assert seen[1][2] == "ml.cli.sft"


def test_validate_posttrain_stage_inputs_rejects_incompatible_export_dir(tmp_path) -> None:
    plan = build_rehearsal_plan(_make_config(tmp_path))
    _write_hf_config(tmp_path / "out" / "local_rehearsal" / "pretrain_check", max_seq_len=512)

    with pytest.raises(SophiaUsageError, match="requires max_seq_len>=4096"):
        rehearsal_mod.validate_posttrain_stage_inputs(plan.stages[1])


def test_run_rehearsal_marks_interrupted_stage_consistently(
    monkeypatch,
    tmp_path,
) -> None:
    plan = build_rehearsal_plan(_make_config(tmp_path))
    interrupted: list[bool] = []

    class _HangingPopen:
        def __init__(self, argv: list[str], *, cwd: str, env, stdout, stderr) -> None:
            del argv, cwd, env, stdout, stderr

        def poll(self) -> None:
            return None

        def send_signal(self, _sig: int) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    def _raise_keyboard_interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt()

    def _fake_terminate_process(proc, *, interrupt: bool) -> int:
        del proc
        interrupted.append(bool(interrupt))
        return -2

    monkeypatch.setattr(rehearsal_process.subprocess, "Popen", _HangingPopen)
    monkeypatch.setattr(rehearsal_process.time, "sleep", _raise_keyboard_interrupt)
    monkeypatch.setattr(rehearsal_process, "terminate_process", _fake_terminate_process)

    with pytest.raises(SophiaUsageError, match="interrupted during stage: pretrain_check"):
        run_rehearsal(plan)

    report_path = tmp_path / "out" / "local_rehearsal" / "rehearsal_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert interrupted == [True]
    assert report["status"] == "interrupted"
    assert report["stages"][0]["name"] == "pretrain_check"
    assert report["stages"][0]["status"] == "interrupted"
    assert report["stages"][0]["completed_at"] is not None


def test_resolve_rehearsal_config_normalizes_paths_and_interval() -> None:
    resolved = resolve_rehearsal_config(
        RehearsalConfig(
            output_root=" out/local_rehearsal ",
            device="cpu",
            progress_interval_seconds=-1.0,
        )
    )

    assert resolved.output_root == "out/local_rehearsal"
    assert resolved.device == "cpu"
    assert resolved.progress_interval_seconds == 0.0


def test_rehearsal_config_from_args_uses_typed_constructor() -> None:
    args = __import__("argparse").Namespace(
        output_root="out/rehearsal",
        device="cpu",
        seed=7,
        overwrite_output_dir=0,
        pretrain_data="dataset/pretrain_tokens",
        posttrain_curriculum_path="dataset/posttrain_length_curriculum.json",
        pretrain_tokens=1024,
        sft_train_data="dataset/sft/train.jsonl",
        sft_eval_data="dataset/sft/val.jsonl",
        sft_test_data="dataset/sft/test.jsonl",
        sft_max_steps=2,
        posttrain_batch_size=2,
        posttrain_eval_steps=2,
        posttrain_report_max_examples_per_split=3,
        posttrain_report_progress_every=11,
        progress_interval_seconds=5.0,
    )

    config = RehearsalConfig.from_args(args)

    assert config.output_root == "out/rehearsal"
    assert config.device == "cpu"
    assert config.seed == 7
    assert config.overwrite_output_dir is False
    assert config.posttrain_batch_size == 2
    assert config.posttrain_report_progress_every == 11


def test_rehearsal_config_from_args_accepts_mapping() -> None:
    config = RehearsalConfig.from_args(
        {
            "output_root": "out/rehearsal",
            "device": "cpu",
            "seed": 3,
            "overwrite_output_dir": 0,
            "pretrain_data": "dataset/pretrain_tokens",
            "posttrain_curriculum_path": "dataset/posttrain_length_curriculum.json",
            "posttrain_report_progress_every": 17,
        }
    )

    assert config.device == "cpu"
    assert config.seed == 3
    assert config.posttrain_report_progress_every == 17
