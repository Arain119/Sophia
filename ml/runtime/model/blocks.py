from __future__ import annotations

import torch
from torch import nn

from ml.runtime.model.attention import SophiaKDA, SophiaMLA
from ml.runtime.model.config import ModelArgs
from ml.runtime.model.ops import RMSNorm
from ml.runtime.model.runtime_linear import RuntimeLinear


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden = int(args.ffn_hidden)
        self.gate_softcap = float(args.situ_gate_softcap)
        self.up_softcap = float(args.situ_up_softcap)
        self.gate_up_proj = RuntimeLinear(
            int(args.dim), 2 * self.hidden, bias=False
        )
        self.down_proj = RuntimeLinear(self.hidden, int(args.dim), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.hidden, dim=-1)
        bounded_gate = self.gate_softcap * torch.tanh(gate / self.gate_softcap)
        bounded_up = self.up_softcap * torch.tanh(up / self.up_softcap)
        return self.down_proj(bounded_gate * torch.sigmoid(gate) * bounded_up)


class AttentionResidualMixer(nn.Module):
    """Content-dependent mixing over completed depth blocks and a partial block."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.query = RuntimeLinear(args.dim, 1, bias=False)

    def forward(
        self,
        partial_block: torch.Tensor,
        completed_blocks: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat((completed_blocks, partial_block.unsqueeze(2)), dim=2)
        keys = self.norm(values)
        score_weight = self.query.weight.squeeze(0).float()
        scores = torch.einsum("bsth,h->bst", keys.float(), score_weight)
        weights = scores.softmax(dim=2).unsqueeze(-1).to(dtype=values.dtype)
        return (weights * values).sum(dim=2).to(dtype=values.dtype)


class SophiaBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.layer_type = args.layer_type(self.layer_idx)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = (
            SophiaMLA(args) if self.layer_type == "mla" else SophiaKDA(args)
        )
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn = FeedForward(args)
        self.attn_res_block_size = int(args.attn_res_block_size)
        self.attn_residual = AttentionResidualMixer(args)
        self.ffn_residual = AttentionResidualMixer(args)

    def forward(
        self,
        partial_block: torch.Tensor,
        completed_blocks: torch.Tensor,
        *,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.attn_residual(partial_block, completed_blocks)
        if self.layer_idx % self.attn_res_block_size == 0:
            completed_blocks = torch.cat(
                (completed_blocks, partial_block.unsqueeze(2)), dim=2
            )
            partial_block = self.attn(
                self.attn_norm(attn_input), start_pos=int(start_pos)
            )
        else:
            partial_block = partial_block + self.attn(
                self.attn_norm(attn_input), start_pos=int(start_pos)
            )
        ffn_input = self.ffn_residual(partial_block, completed_blocks)
        partial_block = partial_block + self.ffn(self.ffn_norm(ffn_input))
        return partial_block, completed_blocks


__all__ = ["AttentionResidualMixer", "FeedForward", "SophiaBlock"]
