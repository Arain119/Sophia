from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from ml.runtime.model.runtime_linear import RuntimeLinear
from ml.runtime.model.attention import Attention
from ml.runtime.model.config import ModelArgs
from ml.runtime.model.ops import RMSNorm


class FeedForward(nn.Module):
    """Standard SwiGLU feed-forward used by the decoder."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden = int(args.ffn_hidden)
        dim = int(args.dim)
        self.hidden = int(hidden)
        self.gate_up_proj = RuntimeLinear(dim, 2 * hidden, bias=False)
        self.down_proj = RuntimeLinear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.hidden, dim=-1)
        gated = functional.silu(gate) * up
        return self.down_proj(gated)


class TransformerBlock(nn.Module):
    """Single transformer block with exact attention and SwiGLU FFN."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn = FeedForward(args)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x), freqs_cis, start_pos=start_pos)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


__all__ = ["TransformerBlock"]
