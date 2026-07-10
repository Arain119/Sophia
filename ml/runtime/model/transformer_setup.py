from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn

from ml.runtime.model.runtime_linear import RuntimeLinear
from ml.runtime.model.blocks import TransformerBlock
from ml.runtime.model.config import ModelArgs
from ml.runtime.model.ops import RMSNorm, StandardLogitMixer
from ml.runtime.model.runtime_host import TransformerRuntime
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
    dropout: nn.Dropout
    gradient_checkpointing: bool
    gradient_checkpointing_exclude_first: int
    gradient_checkpointing_exclude_last: int
    _prefix_cache_namespace: str

    def register_buffer(
        self,
        name: str,
        tensor: torch.Tensor,
        persistent: bool = True,
    ) -> None: ...


def tie_word_embeddings(transformer: _TransformerSetup) -> None:
    if transformer.output.bias is not None:
        raise ValueError("tied word embeddings require a bias-free output projection")
    if tuple(transformer.tok_embeddings.weight.shape) != tuple(transformer.output.weight.shape):
        raise ValueError(
            "tied word embeddings require matching embedding/output shapes; "
            f"got {tuple(transformer.tok_embeddings.weight.shape)!r} vs "
            f"{tuple(transformer.output.weight.shape)!r}"
        )
    transformer.output.weight = transformer.tok_embeddings.weight


def init_weights(model: nn.Module) -> None:
    base_std = 0.02
    layer_count = max(int(getattr(getattr(model, "args", None), "n_layers", 0) or 0), 1)
    residual_out_std = base_std / math.sqrt(2.0 * float(layer_count))
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)

    for layer in getattr(model, "layers", ()):
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "o_proj"):
            nn.init.normal_(attn.o_proj.weight, mean=0.0, std=residual_out_std)
            if getattr(attn.o_proj, "bias", None) is not None:
                nn.init.zeros_(attn.o_proj.bias)
        ffn = getattr(layer, "ffn", None)
        if ffn is not None and hasattr(ffn, "down_proj"):
            nn.init.normal_(ffn.down_proj.weight, mean=0.0, std=residual_out_std)
            if getattr(ffn.down_proj, "bias", None) is not None:
                nn.init.zeros_(ffn.down_proj.bias)


def initialize_transformer(
    transformer: _TransformerSetup,
    args: ModelArgs,
    *,
    precompute_freqs_cis_fn: Callable[..., torch.Tensor],
) -> None:
    transformer.args = args
    transformer.runtime = TransformerRuntime(
        transformer,
        precompute_freqs_cis=precompute_freqs_cis_fn,
        state=TransformerRuntimeState(),
    )
    transformer.runtime_state = transformer.runtime.state

    transformer.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
    transformer.layers = nn.ModuleList(
        [TransformerBlock(args) for _layer_idx in range(args.n_layers)]
    )
    transformer.norm = RMSNorm(args.dim, args.norm_eps)
    transformer.output = RuntimeLinear(
        args.dim,
        args.vocab_size,
        bias=False,
    )
    tie_word_embeddings(transformer)
    transformer.head_mixer = StandardLogitMixer()
    transformer.dropout = nn.Dropout(args.dropout)
    freqs_cis = precompute_freqs_cis_fn(
        args.rope_head_dim,
        args.max_seq_len,
        args.rope_theta,
        original_seq_len=args.original_seq_len,
        rope_factor=args.rope_factor,
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
    )
    transformer.register_buffer("freqs_cis", freqs_cis, persistent=False)
    transformer.gradient_checkpointing = False
    transformer.gradient_checkpointing_exclude_first = 0
    transformer.gradient_checkpointing_exclude_last = 0
    transformer._prefix_cache_namespace = (
        f"{type(transformer).__module__}.{type(transformer).__qualname__}:{uuid.uuid4().hex}"
    )
    init_weights(transformer)


__all__ = ["init_weights", "initialize_transformer", "tie_word_embeddings"]
