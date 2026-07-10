from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from ml.data.token_shards.shard_manifest import write_json_atomic
from ml.training.pretrain.machine_runtime import PretrainMachineRuntime
from ml.tasks.pretrain.observability.support import log_tag
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.tasks.pretrain.run_kinds import is_budgeted_short_pretrain_run
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.ema import ModelEMA
from ml.training.pretrain.artifacts import PRETRAIN_UPDATE_PROFILE_ARTIFACT
from ml.training.pretrain.engine.data.pretrain import build_pretrain_data_iter
from ml.training.pretrain.engine.grad_clip import clip_gradients_
from ml.training.pretrain.engine.step_runner_impl import StepRunner
from ml.training.pretrain.batch_size_autofit import create_optimizer

PROFILE_WARMUP_UPDATES = 1
PROFILE_TIMED_UPDATES = 3


@dataclass(frozen=True)
class UpdateProfileContext:
    args: PretrainRunConfig
    state: PretrainRuntimeState
    machine_runtime: PretrainMachineRuntime
    output_path: str
    batch_size: int
    accumulation_steps: int


@dataclass(frozen=True)
class UpdateProfileRuntime:
    optimizer: torch.optim.Optimizer
    ema: ModelEMA | None
    data_iter: object


@dataclass(frozen=True)
class UpdateBreakdownRow:
    zero_s: float
    fetch_s: float
    micro_s: float
    clip_s: float
    optim_s: float
    ema_s: float
    update_s: float

    def to_payload(self) -> dict[str, float]:
        return {
            "zero_s": float(self.zero_s),
            "fetch_s": float(self.fetch_s),
            "micro_s": float(self.micro_s),
            "clip_s": float(self.clip_s),
            "optim_s": float(self.optim_s),
            "ema_s": float(self.ema_s),
            "update_s": float(self.update_s),
        }


def resolve_update_profile_context(
    *,
    args: PretrainRunConfig,
    output_dir: str,
    device: torch.device,
) -> UpdateProfileContext | None:
    if device.type != "cuda":
        return None
    if str(output_dir).strip() == "":
        return None
    state = PretrainRuntimeState.from_config(args)
    if is_budgeted_short_pretrain_run(args._sophia_run_kind):
        return None
    output_path = os.path.join(str(output_dir), PRETRAIN_UPDATE_PROFILE_ARTIFACT)
    if os.path.exists(output_path):
        return None
    if int(state.batch_size) <= 0 or int(state.accumulation_steps) <= 0:
        return None
    return UpdateProfileContext(
        args=args,
        state=state,
        machine_runtime=PretrainMachineRuntime.from_source(state),
        output_path=str(output_path),
        batch_size=int(state.batch_size),
        accumulation_steps=int(state.accumulation_steps),
    )


def maybe_build_profile_ema(
    *,
    model: torch.nn.Module,
    ctx: UpdateProfileContext,
) -> ModelEMA | None:
    try:
        decay = float(ctx.state.ema_decay)
        interval = max(int(ctx.state.ema_update_interval), 1)
        if 0.0 < decay < 1.0 and int(interval) > 0:
            return ModelEMA(model, decay=float(decay), update_interval=int(interval))
    except Exception:
        return None
    return None


def build_update_profile_runtime(
    *,
    ctx: UpdateProfileContext,
    model: torch.nn.Module,
    manifest_path: str,
    device: torch.device,
    seq_len: int,
) -> UpdateProfileRuntime:
    model.train(True)
    optimizer = create_optimizer(
        model,
        lr=0.0,
        weight_decay=float(ctx.args.weight_decay),
        betas=(
            float(ctx.args.beta1),
            float(ctx.args.beta2),
        ),
        eps=float(ctx.args.adam_eps),
        layerwise_lr_decay=float(ctx.args.layerwise_lr_decay),
        embedding_lr_scale=float(ctx.args.embedding_lr_scale),
        muon_ns_steps=(
            None
            if int(ctx.args.muon_ns_steps) <= 0
            else int(ctx.args.muon_ns_steps)
        ),
        muon_target_rms=(
            None
            if ctx.args.muon_target_rms is None
            else float(ctx.args.muon_target_rms)
        ),
    )
    data_iter = build_pretrain_data_iter(
        manifest_path=str(manifest_path),
        seq_len=int(seq_len),
        batch_size=int(ctx.batch_size),
        seed=int(ctx.args.seed),
        device=device,
        num_workers=int(ctx.machine_runtime.dataloader_num_workers),
        dataloader_prefetch_factor=int(ctx.machine_runtime.dataloader_prefetch_factor),
        dataloader_persistent_workers=(
            None
            if int(ctx.machine_runtime.dataloader_persistent_workers) < 0
            else int(ctx.machine_runtime.dataloader_persistent_workers)
        ),
        shard_preload=int(ctx.machine_runtime.shard_preload),
        shard_preload_bytes=int(ctx.machine_runtime.shard_preload_bytes),
    )
    return UpdateProfileRuntime(
        optimizer=optimizer,
        ema=maybe_build_profile_ema(model=model, ctx=ctx),
        data_iter=data_iter,
    )


def cleanup_update_profile_runtime(
    *,
    model: torch.nn.Module,
    runner: StepRunner,
    runtime: UpdateProfileRuntime,
) -> None:
    del model, runner
    close_fn = getattr(runtime.data_iter, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass
    try:
        runtime.optimizer.zero_grad(set_to_none=True)
    except Exception:
        pass


def _sync_cuda(device: torch.device) -> None:
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass


def collect_update_breakdown_rows(
    *,
    ctx: UpdateProfileContext,
    runtime: UpdateProfileRuntime,
    model: torch.nn.Module,
    runner: StepRunner,
    device: torch.device,
) -> list[UpdateBreakdownRow]:
    rows: list[UpdateBreakdownRow] = []
    total_updates = int(PROFILE_WARMUP_UPDATES) + int(PROFILE_TIMED_UPDATES)
    for update_idx in range(int(total_updates)):
        _sync_cuda(device)
        update_start = time.perf_counter()

        zero_start = time.perf_counter()
        runtime.optimizer.zero_grad(set_to_none=bool(runner.zero_grad_set_to_none))
        _sync_cuda(device)
        zero_s = float(time.perf_counter() - zero_start)

        runner.begin_update(accum_steps=int(ctx.accumulation_steps))

        fetch_s = 0.0
        micro_s = 0.0
        for _ in range(int(ctx.accumulation_steps)):
            fetch_start = time.perf_counter()
            batch = next(runtime.data_iter)
            _sync_cuda(device)
            fetch_s += float(time.perf_counter() - fetch_start)

            micro_start = time.perf_counter()
            runner.run_micro(batch, accum_steps=int(ctx.accumulation_steps))
            _sync_cuda(device)
            micro_s += float(time.perf_counter() - micro_start)

        clip_start = time.perf_counter()
        clip_gradients_(
            model=model,
            optimizer=runtime.optimizer,
            grad_clip_mode=str(ctx.args.grad_clip_mode),
            max_grad_norm=float(ctx.args.max_grad_norm),
            agc_clip=float(ctx.args.agc_clip),
            agc_eps=float(ctx.args.agc_eps),
            agc_exclude_bias_and_norm=bool(
                int(ctx.args.agc_exclude_bias_and_norm) == 1
            ),
        )
        _sync_cuda(device)
        clip_s = float(time.perf_counter() - clip_start)

        optim_start = time.perf_counter()
        runtime.optimizer.step()
        _sync_cuda(device)
        optim_s = float(time.perf_counter() - optim_start)

        ema_s = 0.0
        if runtime.ema is not None:
            ema_start = time.perf_counter()
            runtime.ema.update()
            _sync_cuda(device)
            ema_s = float(time.perf_counter() - ema_start)

        _sync_cuda(device)
        update_s = float(time.perf_counter() - update_start)
        if update_idx >= int(PROFILE_WARMUP_UPDATES):
            rows.append(
                UpdateBreakdownRow(
                    zero_s=float(zero_s),
                    fetch_s=float(fetch_s),
                    micro_s=float(micro_s),
                    clip_s=float(clip_s),
                    optim_s=float(optim_s),
                    ema_s=float(ema_s),
                    update_s=float(update_s),
                )
            )
    return rows


def average_update_breakdown(rows: list[UpdateBreakdownRow]) -> dict[str, float]:
    if not rows:
        return {}
    payload_rows = tuple(row.to_payload() for row in rows)
    keys = tuple(payload_rows[0].keys())
    return {
        key: float(sum(float(row.get(key, 0.0) or 0.0) for row in payload_rows))
        / float(len(payload_rows))
        for key in keys
    }


def build_update_breakdown_payload(
    *,
    ctx: UpdateProfileContext,
    rows: list[UpdateBreakdownRow],
    seq_len: int,
    device: torch.device,
    runner: StepRunner,
) -> dict[str, object]:
    avg = average_update_breakdown(rows)
    tokens_per_update = (
        int(ctx.batch_size) * int(seq_len) * int(ctx.accumulation_steps)
    )
    update_s = float(avg.get("update_s", 0.0) or 0.0)
    tokens_per_sec = (
        float(tokens_per_update) / float(update_s) if update_s > 0.0 else 0.0
    )
    machine_runtime = ctx.machine_runtime
    return {
        "time": float(time.time()),
        "device": str(device),
        "seq_len": int(seq_len),
        "batch_size": int(ctx.batch_size),
        "accumulation_steps": int(ctx.accumulation_steps),
        "tokens_per_update": int(tokens_per_update),
        "tokens_per_sec_est": float(tokens_per_sec),
        "runner": f"{runner.__class__.__module__}.{runner.__class__.__name__}",
        "dataloader": {
            "num_workers": int(machine_runtime.dataloader_num_workers),
            "prefetch_factor": int(machine_runtime.dataloader_prefetch_factor),
            "persistent_workers": int(
                machine_runtime.dataloader_persistent_workers
            ),
            "shard_preload": int(machine_runtime.shard_preload),
            "shard_preload_bytes": int(machine_runtime.shard_preload_bytes),
        },
        "optimizer": {
            "kind": "torch_muon_hybrid",
            "muon_ns_steps": int(ctx.args.muon_ns_steps),
            "muon_target_rms": (
                None
                if ctx.args.muon_target_rms is None
                else float(ctx.args.muon_target_rms)
            ),
            "embedding_lr_scale": float(ctx.args.embedding_lr_scale),
        },
        "avg_update_breakdown_s": dict(avg),
        "n_timed_updates": int(len(rows)),
    }


def maybe_profile_update_breakdown(
    *,
    args: PretrainRunConfig,
    model: torch.nn.Module,
    runner: StepRunner,
    output_dir: str,
    manifest_path: str,
    device: torch.device,
    seq_len: int,
) -> None:
    ctx = resolve_update_profile_context(
        args=args,
        output_dir=output_dir,
        device=device,
    )
    if ctx is None:
        return
    runtime = build_update_profile_runtime(
        ctx=ctx,
        model=model,
        manifest_path=manifest_path,
        device=device,
        seq_len=int(seq_len),
    )
    try:
        rows = collect_update_breakdown_rows(
            ctx=ctx,
            runtime=runtime,
            model=model,
            runner=runner,
            device=device,
        )
    finally:
        cleanup_update_profile_runtime(
            model=model,
            runner=runner,
            runtime=runtime,
        )

    if not rows:
        return

    payload = build_update_breakdown_payload(
        ctx=ctx,
        rows=rows,
        seq_len=int(seq_len),
        device=device,
        runner=runner,
    )
    avg = dict(payload.get("avg_update_breakdown_s", {}))
    tps = float(payload.get("tokens_per_sec_est", 0.0) or 0.0)
    update_s = float(avg.get("update_s", 0.0) or 0.0)

    try:
        write_json_atomic(str(ctx.output_path), payload)
        print(
            f"{log_tag()} update_profile | toks/s~{tps:.0f} "
            f"update_s={update_s:.4f} "
            f"micro_s={float(avg.get('micro_s', 0.0) or 0.0):.4f} "
            f"optim_s={float(avg.get('optim_s', 0.0) or 0.0):.4f} "
            f"ema_s={float(avg.get('ema_s', 0.0) or 0.0):.4f}",
            flush=True,
        )
    except Exception as exc:
        print(f"[WARN] unable to write {PRETRAIN_UPDATE_PROFILE_ARTIFACT}: {exc}", flush=True)


__all__ = [
    "PROFILE_TIMED_UPDATES",
    "PROFILE_WARMUP_UPDATES",
    "UpdateBreakdownRow",
    "UpdateProfileContext",
    "UpdateProfileRuntime",
    "average_update_breakdown",
    "build_update_breakdown_payload",
    "build_update_profile_runtime",
    "cleanup_update_profile_runtime",
    "collect_update_breakdown_rows",
    "maybe_profile_update_breakdown",
    "resolve_update_profile_context",
]
