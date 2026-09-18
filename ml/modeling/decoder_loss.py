from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import importlib

import torch
from torch import nn

from ml.modeling.text.loss_stats import mean_loss_from_sum_and_count
from ml.modeling.text.loss_stats import shifted_loss_sum_and_count
from ml.modeling.decoder_types import DecoderConfig

def loss_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return shifted_loss_sum_and_count(
        logits,
        labels,
        label_offset=label_offset,
        ignore_index=-100,
    )


def mean_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_offset: int = 0,
) -> torch.Tensor:
    loss_sum, count = loss_stats(logits, labels, label_offset=label_offset)
    return mean_loss_from_sum_and_count(
        loss_sum=loss_sum,
        count=count,
        reference=logits,
    )


@lru_cache(maxsize=1)
def _liger_fused_linear_ce_func() -> Callable[..., torch.Tensor]:
    module_name = "liger_kernel.transformers.functional"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "liger_kernel":
            raise RuntimeError(
                "Liger fused linear cross entropy requires liger_kernel with "
                "liger_fused_linear_cross_entropy"
            ) from exc
        raise
    fn = getattr(module, "liger_fused_linear_cross_entropy", None)
    if fn is None:
        raise RuntimeError(f"{module_name}.liger_fused_linear_cross_entropy is unavailable")
    return fn


@lru_cache(maxsize=1)
def _liger_graph_safe_ops():
    module_name = "liger_kernel.ops.fused_linear_cross_entropy"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "liger_kernel":
            raise RuntimeError(
                "Liger fused linear cross entropy requires liger_kernel"
            ) from exc
        raise
    required = (
        "MAX_FUSED_SIZE",
        "element_mul_kernel",
        "is_hip",
        "liger_cross_entropy_kernel",
        "triton",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"{module_name} is missing required graph-safe operators: {missing}"
        )
    return module


def _liger_linear_ce_loss_chunk_stats(
    hidden_chunk: torch.Tensor,
    chunk_labels: torch.Tensor,
    *,
    norm: nn.Module,
    output: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = getattr(output, "weight", None)
    bias = getattr(output, "bias", None)
    if not torch.is_tensor(weight):
        raise TypeError("liger_linear_ce loss backend requires output.weight")

    flat_labels = chunk_labels.reshape(-1).to(dtype=torch.long)
    keep = flat_labels != -100
    count = keep.sum().to(dtype=torch.float32)

    hidden_norm = norm(hidden_chunk).reshape(-1, int(hidden_chunk.size(-1)))
    loss_sum = _call_liger_fused_linear_ce(
        hidden_norm,
        weight,
        flat_labels,
        bias=bias,
    )
    if not torch.is_tensor(loss_sum):
        raise RuntimeError("liger_fused_linear_cross_entropy must return a tensor loss")
    return loss_sum.float(), count


def _reference_linear_ce_loss_chunk_stats(
    hidden_chunk: torch.Tensor,
    chunk_labels: torch.Tensor,
    *,
    norm: nn.Module,
    output: nn.Module,
    tile: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same statistic as the fused kernel, in plain PyTorch.

    Liger's fused linear cross entropy is a Triton kernel, so on a device
    without one the model cannot score its own loss at all -- which is exactly
    what a host wants to do when the accelerator is busy training. Every other
    device-specific path in the model already has a reference implementation to
    fall back to; this is the one that did not.

    Logits are materialised a tile of rows at a time because the vocabulary is
    65,536 wide and a whole chunk at once is gigabytes for no reason.
    """
    weight = getattr(output, "weight", None)
    bias = getattr(output, "bias", None)
    if not torch.is_tensor(weight):
        raise TypeError("reference linear_ce loss backend requires output.weight")

    flat_labels = chunk_labels.reshape(-1).to(dtype=torch.long)
    keep = flat_labels != -100
    count = keep.sum().to(dtype=torch.float32)

    hidden_norm = norm(hidden_chunk).reshape(-1, int(hidden_chunk.size(-1)))
    weight_f32 = weight.float()
    bias_f32 = None if bias is None else bias.float()
    loss_sum = hidden_norm.new_zeros((), dtype=torch.float32)
    rows = int(hidden_norm.size(0))
    for start in range(0, rows, int(tile)):
        end = min(start + int(tile), rows)
        logits = torch.nn.functional.linear(
            hidden_norm[start:end].float(), weight_f32, bias_f32
        )
        loss_sum = loss_sum + torch.nn.functional.cross_entropy(
            logits,
            flat_labels[start:end],
            ignore_index=-100,
            reduction="sum",
        )
    return loss_sum.float(), count


class _GraphSafeLinearCE(torch.autograd.Function):
    _TILE = 128
    _CUDA_CHUNK_SIZE = 4096

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.is_cuda:
            ops = _liger_graph_safe_ops()
            token_count = int(hidden.size(0))
            vocab_size = int(weight.shape[0])
            block_size = min(
                int(ops.MAX_FUSED_SIZE),
                int(ops.triton.next_power_of_2(vocab_size)),
            )
            # The release graph is bounded by the 4096-token context. A single
            # full-width logits GEMM maximizes M without introducing a second
            # sequence-length capability or an unbounded workspace.
            chunk_size = min(int(token_count), _GraphSafeLinearCE._CUDA_CHUNK_SIZE)
            chunk_count = int(ops.triton.cdiv(int(token_count), chunk_size))
            has_bias = bool(bias.numel())
            grad_hidden = torch.empty_like(hidden)
            grad_weight = torch.empty_like(weight)
            grad_bias = torch.empty_like(bias) if has_bias else None
            loss_per_token = torch.zeros(
                int(token_count),
                dtype=torch.float32,
                device=hidden.device,
            )

            for chunk_index in range(chunk_count):
                start = chunk_index * chunk_size
                end = min(start + chunk_size, int(token_count))
                hidden_chunk = hidden[start:end]
                logits = hidden_chunk @ weight.t()
                if has_bias:
                    logits = logits + bias
                logits = logits.contiguous()
                label_chunk = labels[start:end].contiguous()
                loss_chunk = loss_per_token[start:end]
                row_count = int(logits.shape[0])
                ops.liger_cross_entropy_kernel[(row_count,)](
                    X_ptr=logits,
                    X_stride=logits.stride(-2),
                    Y_ptr=label_chunk,
                    Y_stride=label_chunk.stride(-1),
                    weight_ptr=None,
                    loss_ptr=loss_chunk,
                    z_loss_ptr=None,
                    loss_stride=loss_chunk.stride(-1),
                    token_accuracy_ptr=None,
                    token_accuracy_stride=0,
                    predicted_tokens_ptr=None,
                    predicted_tokens_stride=0,
                    n_cols=vocab_size,
                    n_non_ignore=int(token_count),
                    sum_non_ignore_weight=int(token_count),
                    weight_sum=0.0,
                    ignore_index=-100,
                    lse_square_scale=0.0,
                    label_smoothing=0.0,
                    reduction="sum",
                    softcap=None,
                    RETURN_Z_LOSS=False,
                    RETURN_TOKEN_ACCURACY=False,
                    RETURN_PREDICTED_TOKENS=False,
                    HAS_WEIGHT=False,
                    HAS_SOFTCAPPING=False,
                    HAS_GRADIENTS=True,
                    BLOCK_SIZE=block_size,
                    num_warps=32 if not ops.is_hip() else 16,
                )
                torch.mm(
                    logits,
                    weight,
                    out=grad_hidden[start:end],
                )
                if chunk_index == 0:
                    torch.mm(logits.transpose(0, 1), hidden_chunk, out=grad_weight)
                else:
                    torch.addmm(
                        grad_weight,
                        logits.transpose(0, 1),
                        hidden_chunk,
                        out=grad_weight,
                    )
                if grad_bias is not None:
                    bias_grad = logits.sum(dim=0)
                    if chunk_index == 0:
                        grad_bias.copy_(bias_grad)
                    else:
                        grad_bias.add_(bias_grad)

            saved_bias_grad = (
                grad_bias if grad_bias is not None else hidden.new_empty((0,))
            )
            ctx.save_for_backward(grad_hidden, grad_weight, saved_bias_grad)
            ctx.has_bias = has_bias
            ctx.liger_cuda = True
            return loss_per_token.sum()

        ctx.save_for_backward(hidden, weight, labels, bias)
        ctx.has_bias = bool(bias.numel())
        ctx.liger_cuda = False
        total = hidden.new_zeros((), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, int(hidden.size(0)), _GraphSafeLinearCE._TILE):
                logits = torch.nn.functional.linear(
                    hidden[start : start + _GraphSafeLinearCE._TILE],
                    weight,
                    bias if ctx.has_bias else None,
                )
                total = total + torch.nn.functional.cross_entropy(
                    logits,
                    labels[start : start + _GraphSafeLinearCE._TILE],
                    ignore_index=-100,
                    reduction="sum",
                ).float()
        return total

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.liger_cuda:
            saved_hidden_grad, saved_weight_grad, saved_bias_grad = ctx.saved_tensors
            grad_hidden = torch.empty_like(saved_hidden_grad)
            grad_weight = torch.empty_like(saved_weight_grad)
            grad_hidden.copy_(saved_hidden_grad)
            grad_weight.copy_(saved_weight_grad)
            ops = _liger_graph_safe_ops()
            block_size = min(
                int(ops.MAX_FUSED_SIZE),
                int(ops.triton.next_power_of_2(int(grad_hidden.shape[-1]))),
            )
            num_warps = 32 if not ops.is_hip() else 16
            ops.element_mul_kernel[(int(grad_hidden.shape[0]),)](
                grad_hidden,
                grad_hidden.stride(-2),
                grad_output,
                int(grad_hidden.shape[-1]),
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            ops.element_mul_kernel[(int(grad_weight.shape[0]),)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                int(grad_weight.shape[-1]),
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            grad_bias = None
            if ctx.has_bias:
                grad_bias = torch.empty_like(saved_bias_grad)
                grad_bias.copy_(saved_bias_grad)
                ops.element_mul_kernel[(int(grad_bias.shape[0]),)](
                    grad_bias,
                    grad_bias.stride(-1),
                    grad_output,
                    1,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
            return grad_hidden, grad_weight, None, grad_bias

        hidden, weight, labels, bias = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if ctx.has_bias else None
        for start in range(0, int(hidden.size(0)), _GraphSafeLinearCE._TILE):
            hidden_tile = hidden[start : start + _GraphSafeLinearCE._TILE]
            labels_tile = labels[start : start + _GraphSafeLinearCE._TILE]
            logits = torch.nn.functional.linear(
                hidden_tile,
                weight,
                bias if ctx.has_bias else None,
            )
            valid = labels_tile != -100
            probs = torch.softmax(logits, dim=-1)
            safe_labels = labels_tile.clamp_min(0).unsqueeze(1)
            probs.scatter_add_(
                1,
                safe_labels,
                -valid.to(dtype=probs.dtype).unsqueeze(1),
            )
            probs.mul_(valid.unsqueeze(1))
            torch.mm(
                probs,
                weight,
                out=grad_hidden[start : start + _GraphSafeLinearCE._TILE],
            )
            torch.addmm(
                grad_weight,
                probs.transpose(0, 1),
                hidden_tile,
                out=grad_weight,
            )
            if grad_bias is not None:
                grad_bias.add_(probs.sum(dim=0))
        scale = grad_output.to(dtype=grad_hidden.dtype)
        return grad_hidden * scale, grad_weight * scale, None, (
            None if grad_bias is None else grad_bias * scale
        )


@torch.compiler.disable
def _call_liger_fused_linear_ce(
    hidden_norm: torch.Tensor,
    weight: torch.Tensor,
    flat_labels: torch.Tensor,
    *,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if hidden_norm.is_cuda and torch.cuda.is_current_stream_capturing():
        bias_arg = (
            bias if bias is not None else hidden_norm.new_empty((0,))
        )
        return _GraphSafeLinearCE.apply(hidden_norm, weight, flat_labels, bias_arg)
    fn = _liger_fused_linear_ce_func()
    try:
        return fn(
            hidden_norm,
            weight,
            flat_labels,
            bias=bias,
            ignore_index=-100,
            reduction="sum",
        )
    except TypeError as exc:
        raise RuntimeError(
            "installed liger_fused_linear_cross_entropy has an unsupported signature"
        ) from exc


def _loss_chunk_stats(
    hidden_chunk: torch.Tensor,
    chunk_labels: torch.Tensor,
    *,
    norm: nn.Module,
    output: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(
        getattr(output, "_sophia_release_cuda_cache_before_linear_ce", False)
    ):
        output._sophia_release_cuda_cache_before_linear_ce = False
        torch.cuda.empty_cache()
    if not hidden_chunk.is_cuda:
        return _reference_linear_ce_loss_chunk_stats(
            hidden_chunk,
            chunk_labels,
            norm=norm,
            output=output,
        )
    return _liger_linear_ce_loss_chunk_stats(
        hidden_chunk,
        chunk_labels,
        norm=norm,
        output=output,
    )


def chunked_loss_stats_from_hidden(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_offset: int,
    norm: nn.Module,
    output: nn.Module,
    config: DecoderConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_pred_tokens = min(
        max(int(hidden.size(1)) - 1, 0),
        max(int(labels.size(1)) - int(label_offset) - 1, 0),
    )
    zero = hidden.new_zeros(())
    if max_pred_tokens <= 0:
        return zero, zero

    loss_sum = zero
    count = zero
    chunk_size = max(int(config.loss_chunk_size), 0)
    if chunk_size <= 0:
        chunk_size = int(max_pred_tokens)
    target_base = int(label_offset) + 1
    for token_start in range(0, max_pred_tokens, chunk_size):
        token_end = min(token_start + chunk_size, max_pred_tokens)
        chunk_labels = labels[
            :,
            target_base + token_start : target_base + token_end,
        ].contiguous()
        chunk_loss_sum, chunk_count = _loss_chunk_stats(
            hidden[:, token_start:token_end],
            chunk_labels,
            norm=norm,
            output=output,
        )
        loss_sum = loss_sum + chunk_loss_sum
        count = count + chunk_count
    return loss_sum, count


__all__ = [
    "chunked_loss_stats_from_hidden",
    "loss_stats",
    "mean_cross_entropy_loss",
]
