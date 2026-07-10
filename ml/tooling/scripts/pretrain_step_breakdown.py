from __future__ import annotations

import argparse
import gc
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ml.modeling import SophiaDecoderConfig
from ml.runtime.torch_env import setup_seed, setup_torch_backends
from ml.tasks.pretrain.pipeline import build_pretrain_args
from ml.tasks.pretrain.run_spec_builder import build_pretrain_run_spec
from ml.training.pretrain.engine.grad_clip import clip_gradients_
from ml.training.pretrain.engine.step_runner_impl import build_step_runner
from ml.training.pretrain.model_setup import load_or_init_model
from ml.training.pretrain.optimizer import create_torch_muon_optimizer
from ml.training.pretrain.profiles import resolve_pretrain_profile


DEFAULT_CASES = "0"
TORCH_INDUCTOR_COMPILE_MODE_ENV = "SOPHIA_TORCH_INDUCTOR_COMPILE_MODE"
TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV = "SOPHIA_TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"


@dataclass(frozen=True)
class StepBreakdownCase:
    loss_chunk_size: int = 0

    @property
    def name(self) -> str:
        chunk = int(self.loss_chunk_size)
        return "chunk0" if chunk <= 0 else f"chunk{chunk}"


def parse_cases(value: str) -> tuple[StepBreakdownCase, ...]:
    raw = str(value or DEFAULT_CASES).strip()
    cases: list[StepBreakdownCase] = []
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        cases.append(StepBreakdownCase(loss_chunk_size=int(part)))
    if not cases:
        raise ValueError("at least one benchmark case is required")
    return tuple(cases)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _temporary_env_values(values: dict[str, object]):
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        text = str(value).strip()
        if text:
            os.environ[str(key)] = text
        else:
            os.environ.pop(str(key), None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_decoder_config(
    *,
    args,
    seq_len: int,
    batch_size: int,
) -> tuple[SophiaDecoderConfig, int]:
    profile = resolve_pretrain_profile(args)
    cfg = profile.model.with_overrides(
        vocab_size=int(profile.model.vocab_size),
        max_seq_len=int(seq_len),
        max_batch_size=int(batch_size),
    ).to_config()
    decoder_config = SophiaDecoderConfig.from_object(cfg)
    decoder_config.return_logits_in_train = False
    decoder_config.loss_chunk_size = int(getattr(args, "loss_chunk_size", 0) or 0)
    return decoder_config, int(profile.model.vocab_size)


def _make_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:
    return create_torch_muon_optimizer(
        model,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        betas=(float(args.beta1), float(args.beta2)),
        eps=float(args.adam_eps),
        layerwise_lr_decay=float(args.layerwise_lr_decay),
        embedding_lr_scale=float(args.embedding_lr_scale),
        muon_ns_steps=int(args.muon_ns_steps),
        muon_target_rms=float(args.muon_target_rms),
    )


def _run_update(
    *,
    model: torch.nn.Module,
    runner,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    args,
    accumulation_steps: int,
    include_optimizer: bool,
) -> dict[str, float | None]:
    optimizer.zero_grad(set_to_none=bool(runner.zero_grad_set_to_none))

    _sync()
    start = time.perf_counter()
    loss_value = None
    for _ in range(int(accumulation_steps)):
        out = runner.run_micro(batch, accum_steps=int(accumulation_steps))
        if torch.is_tensor(out):
            loss_value = float(out.detach().float().item())
    _sync()
    micro_s = float(time.perf_counter() - start)

    _sync()
    start = time.perf_counter()
    grad_norm = clip_gradients_(
        model=model,
        optimizer=optimizer,
        grad_clip_mode=str(args.grad_clip_mode),
        max_grad_norm=float(args.max_grad_norm),
        agc_clip=float(args.agc_clip),
        agc_eps=float(args.agc_eps),
        agc_exclude_bias_and_norm=bool(int(args.agc_exclude_bias_and_norm) == 1),
    )
    _sync()
    clip_s = float(time.perf_counter() - start)

    optim_s = 0.0
    if bool(include_optimizer):
        _sync()
        start = time.perf_counter()
        optimizer.step()
        _sync()
        optim_s = float(time.perf_counter() - start)

    grad_norm_value = None
    if grad_norm is not None:
        grad_norm_value = float(grad_norm.detach().float().item())
    return {
        "micro_s": float(micro_s),
        "clip_s": float(clip_s),
        "optim_s": float(optim_s),
        "total_s": float(micro_s + clip_s + optim_s),
        "loss": loss_value,
        "grad_norm": grad_norm_value,
    }


def benchmark_case(
    case: StepBreakdownCase,
    *,
    data_path: str,
    seq_len: int,
    batch_size: int,
    accumulation_steps: int,
    warmup_updates: int,
    timed_updates: int,
    include_optimizer: bool,
    device: str,
    seed: int,
    step_execution_backend: str = "eager",
    loss_chunk_size: int | None = None,
    compile_mode: str = "",
    disable_cudagraphs: int = -1,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    setup_seed(int(seed))
    setup_torch_backends()
    torch.cuda.set_device(torch.device(str(device)))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    args = build_pretrain_args(data_path=str(data_path))
    resolved_loss_chunk_size = (
        int(case.loss_chunk_size)
        if loss_chunk_size is None
        else int(loss_chunk_size)
    )
    args.batch_size = int(batch_size)
    args.accumulation_steps = int(accumulation_steps)
    args.target_tokens_per_microbatch = int(batch_size) * int(seq_len)
    args.seq_len = int(seq_len)
    args.max_seq_len = int(seq_len)
    args.loss_chunk_size = int(resolved_loss_chunk_size)
    args.step_execution_backend = str(step_execution_backend)

    env_updates: dict[str, object] = {}
    if str(compile_mode).strip():
        env_updates[TORCH_INDUCTOR_COMPILE_MODE_ENV] = str(compile_mode).strip()
    if int(disable_cudagraphs) >= 0:
        env_updates[TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV] = int(disable_cudagraphs)

    with _temporary_env_values(env_updates):
        decoder_config, vocab_size = _build_decoder_config(
            args=args,
            seq_len=int(seq_len),
            batch_size=int(batch_size),
        )
        build_start = time.perf_counter()
        model = load_or_init_model(
            args=args,
            decoder_config=decoder_config,
            device=torch.device(str(device)),
            base_dtype=torch.bfloat16,
        )
        model.train()
        runner = build_step_runner(
            model=model,
            base_dtype=torch.bfloat16,
            execution_plan=build_pretrain_run_spec(args).step_execution,
            token_weighted=False,
        )
        optimizer = _make_optimizer(args, model)
        _sync()
        build_s = float(time.perf_counter() - build_start)

        input_ids = torch.randint(
            0,
            int(vocab_size),
            (int(batch_size), int(seq_len)),
            device=torch.device(str(device)),
            dtype=torch.long,
        )
        batch = {"input_ids": input_ids, "labels": input_ids.clone()}

        for _ in range(max(int(warmup_updates), 0)):
            _run_update(
                model=model,
                runner=runner,
                optimizer=optimizer,
                batch=batch,
                args=args,
                accumulation_steps=int(accumulation_steps),
                include_optimizer=bool(include_optimizer),
            )

        rows = []
        for _ in range(max(int(timed_updates), 1)):
            rows.append(
                _run_update(
                    model=model,
                    runner=runner,
                    optimizer=optimizer,
                    batch=batch,
                    args=args,
                    accumulation_steps=int(accumulation_steps),
                    include_optimizer=bool(include_optimizer),
                )
            )

    total_times = sorted(float(row["total_s"] or 0.0) for row in rows)
    micro_times = sorted(float(row["micro_s"] or 0.0) for row in rows)
    median_total_s = total_times[len(total_times) // 2]
    median_micro_s = micro_times[len(micro_times) // 2]
    tokens_per_update = int(batch_size) * int(seq_len) * int(accumulation_steps)
    return {
        "ok": True,
        "case": asdict(case),
        "name": case.name,
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "accumulation_steps": int(accumulation_steps),
        "tokens_per_update": int(tokens_per_update),
        "loss_chunk_size": int(args.loss_chunk_size),
        "build_s": float(build_s),
        "timed_updates": rows,
        "median_total_s": float(median_total_s),
        "median_micro_s": float(median_micro_s),
        "tokens_per_s": float(tokens_per_update) / max(float(median_total_s), 1e-9),
        "micro_tokens_per_s": float(tokens_per_update) / max(float(median_micro_s), 1e-9),
        "peak_gb": float(torch.cuda.max_memory_allocated()) / float(1024**3),
        "peak_reserved_gb": float(torch.cuda.max_memory_reserved()) / float(1024**3),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "step_execution_backend": str(step_execution_backend),
        "compile_mode": str(compile_mode).strip(),
        "disable_cudagraphs": int(disable_cudagraphs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Break down real pretrain StepRunner micro/clip/optimizer time."
    )
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_tokens")
    parser.add_argument("--cases", type=str, default=DEFAULT_CASES)
    parser.add_argument("--loss_chunk_size", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_updates", type=int, default=1)
    parser.add_argument("--timed_updates", type=int, default=1)
    parser.add_argument("--include_optimizer", type=int, choices=[0, 1], default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step_execution_backend", type=str, default="eager")
    parser.add_argument("--compile_mode", type=str, default="")
    parser.add_argument("--disable_cudagraphs", type=int, choices=[-1, 0, 1], default=-1)
    parser.add_argument("--output_json", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = [
        benchmark_case(
            case,
            data_path=str(args.data_path),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            accumulation_steps=int(args.accumulation_steps),
            warmup_updates=int(args.warmup_updates),
            timed_updates=int(args.timed_updates),
            include_optimizer=bool(int(args.include_optimizer)),
            device=str(args.device),
            seed=int(args.seed),
            step_execution_backend=str(args.step_execution_backend),
            compile_mode=str(args.compile_mode),
            disable_cudagraphs=int(args.disable_cudagraphs),
            loss_chunk_size=(
                None if int(args.loss_chunk_size) < 0 else int(args.loss_chunk_size)
            ),
        )
        for case in parse_cases(str(args.cases))
    ]
    payload = {"kind": "pretrain_step_breakdown", "results": results}
    for row in results:
        print(json.dumps(row, sort_keys=True), flush=True)
    if str(args.output_json).strip():
        out = Path(str(args.output_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
