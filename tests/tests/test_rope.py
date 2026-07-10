import math

import torch

from ml.runtime.model.rope import apply_rotary_emb


def _rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B,S,H,D] or [T,H,D]
    # cos/sin: [S,D/2] or [B,S,D/2] or [T,D/2]
    if cos.dim() == 2:
        if x.dim() == 4:
            cos_u = cos.unsqueeze(0).unsqueeze(2)  # [1,S,1,D/2]
            sin_u = sin.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 3:
            cos_u = cos.unsqueeze(1)  # [T,1,D/2]
            sin_u = sin.unsqueeze(1)
        else:
            raise ValueError(f"invalid x shape: {tuple(x.shape)}")
    elif cos.dim() == 3:
        if x.dim() != 4:
            raise ValueError(f"invalid x shape: {tuple(x.shape)}")
        cos_u = cos.unsqueeze(2)  # [B,S,1,D/2]
        sin_u = sin.unsqueeze(2)
    else:
        raise ValueError(
            f"invalid cos/sin shapes: cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
        )

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos_u - x2 * sin_u, x1 * sin_u + x2 * cos_u], dim=-1)


def test_sophia_rope_matches_reference_2d() -> None:
    bsz, seq_len, nheads, head_dim = 2, 16, 5, 128
    x = torch.randn((bsz, seq_len, nheads, head_dim), dtype=torch.float32)

    theta = torch.rand((seq_len, head_dim // 2), dtype=torch.float32) * (2 * math.pi)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    x2 = apply_rotary_emb(x.clone(), cos, sin)
    ref = _rope_reference(x, cos, sin)
    assert torch.allclose(x2, ref, atol=1e-5, rtol=1e-5)


def test_sophia_rope_matches_reference_3d() -> None:
    bsz, seq_len, nheads, head_dim = 2, 16, 5, 128
    x = torch.randn((bsz, seq_len, nheads, head_dim), dtype=torch.float32)

    max_pos = 64
    theta_table = torch.rand((max_pos, head_dim // 2), dtype=torch.float32) * (2 * math.pi)
    cos_table = torch.cos(theta_table)
    sin_table = torch.sin(theta_table)

    position_ids = torch.randint(0, max_pos, (bsz, seq_len), dtype=torch.long)
    flat = position_ids.reshape(-1)
    cos = cos_table.index_select(0, flat).view(bsz, seq_len, -1)
    sin = sin_table.index_select(0, flat).view(bsz, seq_len, -1)

    x2 = apply_rotary_emb(x.clone(), cos, sin)
    ref = _rope_reference(x, cos, sin)
    assert torch.allclose(x2, ref, atol=1e-5, rtol=1e-5)


def test_sophia_rope_table_path_matches_manual_reference_for_thd_layout() -> None:
    seq_len, nheads, head_dim = 16, 5, 128
    x = torch.randn((seq_len, nheads, head_dim), dtype=torch.float32)
    theta = torch.rand((seq_len, head_dim // 2), dtype=torch.float32) * (2 * math.pi)
    freqs_cis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    x2 = apply_rotary_emb(x.clone(), freqs_cis, interleaved=False)
    ref = torch.view_as_real(
        torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        * torch.view_as_complex(freqs_cis).unsqueeze(1)
    ).flatten(-2)

    assert torch.allclose(x2, ref.type_as(x), atol=1e-5, rtol=1e-5)


def test_sophia_rope_table_interleaved_path_matches_manual_reference_for_thd_layout() -> None:
    seq_len, nheads, head_dim = 16, 5, 128
    x = torch.randn((seq_len, nheads, head_dim), dtype=torch.float32)
    theta = torch.rand((seq_len, head_dim // 2), dtype=torch.float32) * (2 * math.pi)
    freqs_cis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    x2 = apply_rotary_emb(x.clone(), freqs_cis, interleaved=True)
    x_even = x.float()[..., 0::2]
    x_odd = x.float()[..., 1::2]
    rope_dim_half = int(freqs_cis.size(-2))
    x_complex = torch.view_as_complex(
        torch.stack([x_even[..., -rope_dim_half:], x_odd[..., -rope_dim_half:]], dim=-1)
    )
    x_rotated = x_complex * torch.view_as_complex(freqs_cis).unsqueeze(1)
    x_rotated_real = torch.view_as_real(x_rotated)
    x_even[..., -rope_dim_half:] = x_rotated_real[..., 0]
    x_odd[..., -rope_dim_half:] = x_rotated_real[..., 1]
    ref = torch.stack([x_even, x_odd], dim=-1).flatten(-2)

    assert torch.allclose(x2, ref.type_as(x), atol=1e-5, rtol=1e-5)


def test_sophia_rope_table_path_matches_manual_reference_for_bshd_layout() -> None:
    batch, seq_len, nheads, head_dim = 2, 16, 5, 128
    x = torch.randn((batch, seq_len, nheads, head_dim), dtype=torch.float32)
    theta = torch.rand((seq_len, head_dim // 2), dtype=torch.float32) * (2 * math.pi)
    freqs_cis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    x2 = apply_rotary_emb(x.clone(), freqs_cis, interleaved=False)
    ref = torch.view_as_real(
        torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        * torch.view_as_complex(freqs_cis).unsqueeze(0).unsqueeze(2)
    ).flatten(-2)

    assert torch.allclose(x2, ref.type_as(x), atol=1e-5, rtol=1e-5)


def test_sophia_rope_table_path_prefers_thd_layout_when_t_equals_h() -> None:
    seq_len = nheads = 8
    head_dim = 128
    x = torch.randn((seq_len, nheads, head_dim), dtype=torch.float32)
    theta = torch.rand((seq_len, head_dim // 2), dtype=torch.float32) * (2 * math.pi)
    freqs_cis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    x2 = apply_rotary_emb(x.clone(), freqs_cis, interleaved=False)
    ref = torch.view_as_real(
        torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        * torch.view_as_complex(freqs_cis).unsqueeze(1)
    ).flatten(-2)

    assert torch.allclose(x2, ref.type_as(x), atol=1e-5, rtol=1e-5)


def test_sophia_rope_table_interleaved_path_prefers_thd_layout_when_t_equals_h() -> None:
    seq_len = nheads = 8
    head_dim = 128
    x = torch.randn((seq_len, nheads, head_dim), dtype=torch.float32)
    theta = torch.rand((seq_len, head_dim // 2), dtype=torch.float32) * (2 * math.pi)
    freqs_cis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    x2 = apply_rotary_emb(x.clone(), freqs_cis, interleaved=True)
    x_even = x.float()[..., 0::2]
    x_odd = x.float()[..., 1::2]
    rope_dim_half = int(freqs_cis.size(-2))
    x_complex = torch.view_as_complex(
        torch.stack([x_even[..., -rope_dim_half:], x_odd[..., -rope_dim_half:]], dim=-1)
    )
    x_rotated = x_complex * torch.view_as_complex(freqs_cis).unsqueeze(1)
    x_rotated_real = torch.view_as_real(x_rotated)
    x_even[..., -rope_dim_half:] = x_rotated_real[..., 0]
    x_odd[..., -rope_dim_half:] = x_rotated_real[..., 1]
    ref = torch.stack([x_even, x_odd], dim=-1).flatten(-2)

    assert torch.allclose(x2, ref.type_as(x), atol=1e-5, rtol=1e-5)


def test_sophia_rope_table_bfloat16_preserves_dtype_cpu() -> None:
    batch, seq_len, nheads, head_dim = 2, 8, 3, 64
    x = torch.randn((batch, seq_len, nheads, head_dim), dtype=torch.bfloat16)
    theta = torch.rand((seq_len, head_dim // 2), dtype=torch.float32) * (2 * math.pi)
    freqs_cis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    x2 = apply_rotary_emb(x.clone(), freqs_cis, interleaved=True)

    assert x2.dtype == torch.bfloat16
    assert x2.shape == x.shape
