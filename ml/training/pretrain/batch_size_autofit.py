"""
Binary-search batch-size auto-fit via CUDA OOM probing, plus re-exports of the
optimizer/checkpointing setup helpers pretrain callers commonly need alongside it.

This module is intentionally limited to PyTorch to keep imports lightweight. It also
disables optional Transformers backends (TF/Flax/JAX) to prevent accidental
heavy dependency imports in environments where those stacks are not configured.
"""

from __future__ import annotations

import gc
import logging

import torch
from ml.runtime.stack import disable_optional_ml_backends
from ml.training.model_contracts import require_compute_loss
from ml.training.runtime_tools import (
    apply_gradient_checkpointing,
    create_optimizer,
    move_optimizer_state_to_device,
)

disable_optional_ml_backends()

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger("Sophia")
TORCH_MODULE = torch


def _is_cuda_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg and "cuda" in msg


def _reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if key == "step":
                if torch.is_tensor(value):
                    value.zero_()
                else:
                    state["step"] = 0
                continue
            if torch.is_tensor(value):
                value.zero_()


def _clear_cuda_memory() -> None:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def _probe_backward(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    batch_size: int,
) -> bool:
    batch_size = int(batch_size)
    if batch_size <= 0:
        return False
    require_compute_loss(model, context="Autofit backward probe")

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)

    input_ids: torch.Tensor | None = None
    out = None
    loss = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.int32,
        )
        batch: dict[str, torch.Tensor | bool] = {
            "input_ids": input_ids,
            "labels": input_ids,
            "compute_loss": True,
        }
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.cuda.synchronize(device)
        return True
    except BaseException as err:
        if _is_cuda_oom(err):
            return False
        raise
    finally:
        del loss
        del out
        del input_ids
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        _clear_cuda_memory()


def _materialize_optimizer_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    min_batch_size: int,
) -> None:
    if len(optimizer.state) != 0:
        return
    require_compute_loss(model, context="Autofit optimizer-state materialization")

    saved_groups: list[tuple[float, float]] = []
    for group in optimizer.param_groups:
        saved_groups.append(
            (float(group.get("lr", 0.0)), float(group.get("weight_decay", 0.0)))
        )
        group["lr"] = 0.0
        if "weight_decay" in group:
            group["weight_decay"] = 0.0

    input_ids: torch.Tensor | None = None
    out = None
    loss = None
    local_ok = False
    try:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(min_batch_size, seq_len),
            device=device,
            dtype=torch.int32,
        )
        batch: dict[str, torch.Tensor | bool] = {
            "input_ids": input_ids,
            "labels": input_ids,
            "compute_loss": True,
        }
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.cuda.synchronize(device)
        local_ok = True
    except BaseException as err:
        if _is_cuda_oom(err):
            local_ok = False
        else:
            raise
    finally:
        del loss
        del out
        del input_ids

    if not local_ok:
        for group, (lr, wd) in zip(optimizer.param_groups, saved_groups, strict=True):
            group["lr"] = float(lr)
            if "weight_decay" in group:
                group["weight_decay"] = float(wd)
        optimizer.zero_grad(set_to_none=True)
        _clear_cuda_memory()
        raise RuntimeError("Unable to materialize optimizer state: batch_size=1 OOM.")

    try:
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param is None or not hasattr(param, "grad"):
                    continue
                if param.grad is not None:
                    param.grad.detach().zero_()
        optimizer.step()
        _reset_optimizer_state(optimizer)
    finally:
        for group, (lr, wd) in zip(optimizer.param_groups, saved_groups, strict=True):
            group["lr"] = float(lr)
            if "weight_decay" in group:
                group["weight_decay"] = float(wd)
        optimizer.zero_grad(set_to_none=True)
        _clear_cuda_memory()


def auto_fit_batch_size(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    max_batch_size: int = 1024,
    min_batch_size: int = 1,
) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("auto_fit_batch_size requires CUDA.")

    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0 (got {seq_len}).")
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0 (got {vocab_size}).")
    max_batch_size = int(max_batch_size)
    min_batch_size = int(min_batch_size)
    if max_batch_size < min_batch_size:
        raise ValueError(
            f"max_batch_size must be >= min_batch_size (got {max_batch_size} < {min_batch_size})."
        )

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(
            f"device must be CUDA for auto_fit_batch_size (got {device!r})."
        )

    if optimizer is not None:
        _materialize_optimizer_state(
            model=model,
            optimizer=optimizer,
            device=dev,
            seq_len=seq_len,
            vocab_size=vocab_size,
            min_batch_size=min_batch_size,
        )

    low = 0
    high = max_batch_size
    probe = min_batch_size
    while probe <= high:
        ok = _probe_backward(
            model=model,
            optimizer=optimizer,
            device=dev,
            seq_len=seq_len,
            vocab_size=vocab_size,
            batch_size=probe,
        )
        if ok:
            low = probe
            probe = probe * 2
            continue
        high = probe - 1
        break

    while low < high:
        mid = (low + high + 1) // 2
        ok = _probe_backward(
            model=model,
            optimizer=optimizer,
            device=dev,
            seq_len=seq_len,
            vocab_size=vocab_size,
            batch_size=mid,
        )
        if ok:
            low = mid
        else:
            high = mid - 1

    if low < min_batch_size:
        gpu_name = str(torch.cuda.get_device_name(dev))
        raise RuntimeError(
            "No valid batch size found (even batch_size=1 failed).\n"
            "This usually means the active GPU cannot fit the pinned Sophia runtime spec.\n"
            f"gpu={gpu_name}\n"
            f"seq_len={int(seq_len)}\n"
            f"max_batch_size_probe={int(max_batch_size)}\n"
            "Use a larger training GPU for the full spec, or temporarily scale the local validation spec down."
        )
    return int(low)

__all__ = [
    "apply_gradient_checkpointing",
    "auto_fit_batch_size",
    "create_optimizer",
    "move_optimizer_state_to_device",
]
