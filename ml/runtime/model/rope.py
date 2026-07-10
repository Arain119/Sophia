"""
Sophia RoPE (Rotary Position Embedding) implementation.

Implements the default rotary path plus optional scaled-position recipe.
"""

from __future__ import annotations

import math

import torch


def _apply_rotary_emb_cos_sin(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype, device=x.device)
    sin = sin.to(dtype=x.dtype, device=x.device)
    if cos.dim() == 2:
        if x.dim() == 4:
            cos_u = cos.unsqueeze(0).unsqueeze(2)
            sin_u = sin.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 3:
            cos_u = cos.unsqueeze(1)
            sin_u = sin.unsqueeze(1)
        else:
            raise ValueError(f"invalid x shape: {tuple(x.shape)}")
    elif cos.dim() == 3:
        if x.dim() != 4:
            raise ValueError(f"invalid x shape: {tuple(x.shape)}")
        cos_u = cos.unsqueeze(2)
        sin_u = sin.unsqueeze(2)
    else:
        raise ValueError(
            f"invalid cos/sin shapes: cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
        )

    x1, x2 = x.chunk(2, dim=-1)
    if inverse:
        return torch.cat([x1 * cos_u + x2 * sin_u, -x1 * sin_u + x2 * cos_u], dim=-1)
    return torch.cat([x1 * cos_u - x2 * sin_u, x1 * sin_u + x2 * cos_u], dim=-1)


def _normalize_freqs_cis(freqs_cis: torch.Tensor) -> torch.Tensor:
    if freqs_cis.dim() == 3 and int(freqs_cis.size(-1)) == 2:
        return freqs_cis
    raise ValueError(
        "Expected freqs_cis to be [S, D/2, 2] with trailing [cos, sin], "
        f"got shape={tuple(freqs_cis.shape)} dtype={freqs_cis.dtype}"
    )


def _broadcast_freqs_cis(
    x_pairs: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    sequence_dim: int | None = None,
) -> torch.Tensor:
    freqs_cis = _normalize_freqs_cis(freqs_cis)
    seq_len = int(freqs_cis.size(0))
    rope_dim_half = int(freqs_cis.size(-2))
    if int(x_pairs.size(-2)) != rope_dim_half:
        raise ValueError(
            f"freqs_cis last dim ({rope_dim_half}) does not match rotary dim ({int(x_pairs.size(-2))})"
        )

    if x_pairs.ndim == 5:
        if int(x_pairs.size(1)) != seq_len:
            raise ValueError(
                f"freqs_cis seq_len ({seq_len}) does not match x sequence dim ({int(x_pairs.size(1))})"
            )
        return freqs_cis.reshape(1, seq_len, 1, rope_dim_half, 2)

    if x_pairs.ndim == 4:
        if sequence_dim is not None:
            if int(sequence_dim) == 0:
                if int(x_pairs.size(0)) != seq_len:
                    raise ValueError(
                        f"freqs_cis seq_len ({seq_len}) does not match x sequence dim 0 ({int(x_pairs.size(0))})"
                    )
                return freqs_cis.reshape(seq_len, 1, rope_dim_half, 2)
            if int(sequence_dim) == 1:
                if int(x_pairs.size(1)) != seq_len:
                    raise ValueError(
                        f"freqs_cis seq_len ({seq_len}) does not match x sequence dim 1 ({int(x_pairs.size(1))})"
                    )
                return freqs_cis.reshape(1, seq_len, rope_dim_half, 2)
            raise ValueError(f"sequence_dim must be 0 or 1 for 3D rotary inputs, got {sequence_dim}")
        if int(x_pairs.size(0)) == seq_len:
            return freqs_cis.reshape(seq_len, 1, rope_dim_half, 2)
        if int(x_pairs.size(1)) == seq_len:
            return freqs_cis.reshape(1, seq_len, rope_dim_half, 2)
        raise ValueError(
            "freqs_cis seq_len does not match any supported 3D rotary layout: "
            f"x={tuple(x_pairs.shape)} freqs_cis={tuple(freqs_cis.shape)}"
        )

    if x_pairs.ndim == 3:
        if int(x_pairs.size(0)) != seq_len:
            raise ValueError(
                f"freqs_cis seq_len ({seq_len}) does not match x sequence dim 0 ({int(x_pairs.size(0))})"
            )
        return freqs_cis.reshape(seq_len, rope_dim_half, 2)

    raise ValueError(f"Unexpected rotary tensor rank: {tuple(x_pairs.shape)}")


def _apply_rotary_pairs(
    x_pairs: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    inverse: bool,
    sequence_dim: int | None,
) -> torch.Tensor:
    freqs_cis = _broadcast_freqs_cis(
        x_pairs,
        freqs_cis,
        sequence_dim=sequence_dim,
    )
    freqs_cis = freqs_cis.to(dtype=x_pairs.dtype, device=x_pairs.device)
    cos = freqs_cis[..., :1]
    sin = freqs_cis[..., 1:2]
    if bool(inverse):
        sin = -sin
    x0 = x_pairs[..., :1]
    x1 = x_pairs[..., 1:2]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    sin: torch.Tensor | None = None,
    inverse: bool = False,
    interleaved: bool = True,
    sequence_dim: int | None = None,
) -> torch.Tensor:
    if sin is not None:
        return _apply_rotary_emb_cos_sin(x, freqs_cis, sin, inverse=inverse)

    inverse_bool = bool(inverse)
    if interleaved:
        rope_dim_half = int(_normalize_freqs_cis(freqs_cis).size(-2))
        x_pairs = x.unflatten(-1, (int(x.size(-1) // 2), 2))
        x_rotated = _apply_rotary_pairs(
            x_pairs[..., -rope_dim_half:, :],
            freqs_cis,
            inverse=inverse_bool,
            sequence_dim=sequence_dim,
        )
        if int(rope_dim_half) == int(x_pairs.size(-2)):
            x_out = x_rotated
        else:
            x_out = torch.cat([x_pairs[..., :-rope_dim_half, :], x_rotated], dim=-2)
        return x_out.flatten(-2).type_as(x)

    x_flat = x.unflatten(-1, (-1, 2))
    x_out = _apply_rotary_pairs(
        x_flat,
        freqs_cis,
        inverse=inverse_bool,
        sequence_dim=sequence_dim,
    ).flatten(-2)
    return x_out.type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    original_seq_len: int = 0,
    rope_factor: float = 16.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    def find_correction_dim(
        num_rotations: float,
        dim: int,
        base: float,
        max_seq_len: int,
    ) -> float:
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    def find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float,
        max_seq_len: int,
    ) -> tuple[int, int]:
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val: int, max_val: int, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
            max_val - min_val
        )
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast,
            beta_slow,
            dim // 2,
            theta,
            original_seq_len,
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

    positions = torch.arange(end)
    freqs = torch.outer(positions, freqs)
    return torch.stack((torch.cos(freqs), torch.sin(freqs)), dim=-1)

__all__ = ["apply_rotary_emb", "precompute_freqs_cis", "rotate_half"]
