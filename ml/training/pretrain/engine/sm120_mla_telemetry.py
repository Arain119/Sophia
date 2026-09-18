"""SM120 fused MLA logit telemetry."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _max_logit_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        q_ptr,
        k_ptr,
        output_ptr,
        stride_q_batch,
        stride_q_head,
        stride_q_seq,
        stride_q_dim,
        stride_k_batch,
        stride_k_head,
        stride_k_seq,
        stride_k_dim,
        stride_output_head,
        sequence_length,
        num_heads,
        scale,
        BLOCK_M: tl.constexpr,  # noqa: N803
        BLOCK_N: tl.constexpr,  # noqa: N803
        BLOCK_D: tl.constexpr,  # noqa: N803
    ):
        batch_head = tl.program_id(0)
        query_block = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads
        query_positions = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dimensions = tl.arange(0, BLOCK_D)
        q_base = q_ptr + batch * stride_q_batch + head * stride_q_head
        k_base = k_ptr + batch * stride_k_batch + head * stride_k_head
        q = tl.load(
            q_base
            + query_positions[:, None] * stride_q_seq
            + dimensions[None, :] * stride_q_dim,
            mask=(query_positions[:, None] < sequence_length)
            & (dimensions[None, :] < BLOCK_D),
            other=0.0,
        )
        running = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        for key_start in tl.range(
            0, (query_block + 1) * BLOCK_N, BLOCK_N
        ):
            key_positions = key_start + tl.arange(0, BLOCK_N)
            k = tl.load(
                k_base
                + key_positions[:, None] * stride_k_seq
                + dimensions[None, :] * stride_k_dim,
                mask=(key_positions[:, None] < sequence_length)
                & (dimensions[None, :] < BLOCK_D),
                other=0.0,
            )
            logits = tl.dot(
                q,
                tl.trans(k),
                acc=tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
            ) * scale
            valid = (
                (query_positions[:, None] < sequence_length)
                & (key_positions[None, :] < sequence_length)
                & (key_positions[None, :] <= query_positions[:, None])
            )
            logits = tl.where(valid, logits, float("-inf"))
            running = tl.maximum(running, tl.max(logits, axis=1))
        tl.atomic_max(
            output_ptr + head * stride_output_head,
            tl.max(running, axis=0),
        )

    return triton, kernel


@torch.library.custom_op(
    "sophia::mla_attention_logit_max_sm120",
    mutates_args={"output"},
    device_types="cuda",
)
def mla_attention_logit_max_sm120(
    query: torch.Tensor,
    key: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 4:
        raise RuntimeError("SM120 MLA telemetry requires B,H,T,D tensors")
    if int(query.size(0)) != 1 or tuple(query.shape) != tuple(key.shape):
        raise RuntimeError("SM120 MLA telemetry requires matching B1 Q/K tensors")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or int(query.size(-1)) != 128
    ):
        raise RuntimeError("SM120 MLA telemetry requires BF16 head_dim=128")
    if (
        output.ndim != 1
        or int(output.size(0)) != int(query.size(1))
        or output.dtype != torch.float32
        or output.device != query.device
    ):
        raise RuntimeError("SM120 MLA telemetry output shape mismatch")
    triton, kernel = _max_logit_kernel()
    sequence_length = int(query.size(2))
    grid = (int(query.size(0)) * int(query.size(1)), triton.cdiv(sequence_length, 128))
    kernel[grid](
        query,
        key,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        output.stride(0),
        sequence_length,
        int(query.size(1)),
        float(query.size(-1) ** -0.5),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_D=128,
        num_warps=8,
        num_stages=2,
    )


@mla_attention_logit_max_sm120.register_fake
def _mla_attention_logit_max_sm120_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del query, key, output
    return None


@torch.no_grad()
def record_mla_attention_logit_max_sm120(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
) -> None:
    mla_attention_logit_max_sm120(query, key, module.attention_logit_max)


__all__ = [
    "mla_attention_logit_max_sm120",
    "record_mla_attention_logit_max_sm120",
]
