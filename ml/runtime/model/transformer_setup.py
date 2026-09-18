from __future__ import annotations

import math
from typing import Protocol

from torch import nn

from ml.runtime.model.attention import CausalDepthwiseConv1d
from ml.runtime.model.blocks import AttentionResidualMixer, SophiaBlock
from ml.runtime.model.config import ModelArgs
from ml.runtime.model.ops import RMSNorm, StandardLogitMixer
from ml.runtime.model.runtime_host import TransformerRuntime
from ml.runtime.model.runtime_linear import RuntimeLinear
from ml.runtime.model.state import TransformerRuntimeState


class _TransformerSetup(Protocol):
    args: ModelArgs
    runtime: TransformerRuntime
    runtime_state: TransformerRuntimeState
    tok_embeddings: nn.Embedding
    layers: nn.ModuleList
    norm: RMSNorm
    output: RuntimeLinear
    head_mixer: nn.Module
    output_attn_residual: AttentionResidualMixer
    dropout: nn.Dropout
    gradient_checkpointing: bool
    gradient_checkpointing_exclude_first: int
    gradient_checkpointing_exclude_last: int


def tie_word_embeddings(transformer: _TransformerSetup) -> None:
    if tuple(transformer.tok_embeddings.weight.shape) != tuple(transformer.output.weight.shape):
        raise ValueError("tied embedding and output projection shapes must match")
    transformer.output.weight = transformer.tok_embeddings.weight


def init_weights(model: nn.Module) -> None:
    base_std = float(getattr(getattr(model, "args", None), "initializer_range", 0.02))
    if not math.isfinite(base_std) or base_std <= 0.0:
        raise ValueError(f"initializer_range must be finite and > 0, got {base_std}")
    layer_count = max(int(getattr(getattr(model, "args", None), "n_layers", 1)), 1)
    residual_std = base_std / math.sqrt(2.0 * layer_count)
    for module in model.modules():
        if isinstance(module, (nn.Linear, RuntimeLinear)):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)
        elif isinstance(module, CausalDepthwiseConv1d):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)

    for layer in getattr(model, "layers", ()):
        nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=residual_std)
        nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=residual_std)
        nn.init.zeros_(layer.attn_residual.query.weight)
        nn.init.zeros_(layer.ffn_residual.query.weight)
    output_attn_residual = getattr(model, "output_attn_residual", None)
    if output_attn_residual is not None:
        nn.init.zeros_(output_attn_residual.query.weight)


def initialize_transformer(transformer: _TransformerSetup, args: ModelArgs) -> None:
    transformer.args = args
    transformer.runtime = TransformerRuntime(
        transformer,
        state=TransformerRuntimeState(),
    )
    transformer.runtime_state = transformer.runtime.state
    transformer.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
    transformer.layers = nn.ModuleList(
        [SophiaBlock(args, layer_idx) for layer_idx in range(args.n_layers)]
    )
    transformer.norm = RMSNorm(args.dim, args.norm_eps)
    transformer.output_attn_residual = AttentionResidualMixer(args)
    transformer.output = RuntimeLinear(args.dim, args.vocab_size, bias=False)
    tie_word_embeddings(transformer)
    transformer.head_mixer = StandardLogitMixer()
    transformer.dropout = nn.Dropout(float(args.dropout))
    transformer.gradient_checkpointing = False
    transformer.gradient_checkpointing_exclude_first = 0
    transformer.gradient_checkpointing_exclude_last = 0
    init_weights(transformer)


__all__ = ["init_weights", "initialize_transformer", "tie_word_embeddings"]
