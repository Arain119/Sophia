from __future__ import annotations

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import CausalBias, causal_lower_right
from torch.nn.functional import scaled_dot_product_attention

from ml.runtime.model.runtime_linear import RuntimeLinear
from ml.runtime.model.rope import apply_rotary_emb, precompute_freqs_cis
from ml.runtime.model.config import ModelArgs
from ml.runtime.model.ops import RMSNorm


class Attention(nn.Module):
    """
    Exact causal GQA attention used by the decoder.

    CUDA training uses PyTorch SDPA Flash on the supported single-GPU release
    path.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = int(args.dim)
        self.n_heads = int(args.n_heads)
        self.num_key_value_heads = int(args.num_key_value_heads)
        self.head_dim = int(args.head_dim)
        self.max_seq_len = int(args.max_seq_len)
        self.original_seq_len = int(args.original_seq_len)
        self.rope_theta = float(args.rope_theta)
        self.rope_factor = float(args.rope_factor)
        self.beta_fast = int(args.beta_fast)
        self.beta_slow = int(args.beta_slow)
        self.rope_head_dim = int(args.rope_head_dim)
        self.use_cache = bool(args.use_cache)
        self.use_qk_norm = bool(args.use_qk_norm)

        self.q_out_features = self.n_heads * self.head_dim
        self.kv_out_features = self.num_key_value_heads * self.head_dim
        self.qkv_proj = RuntimeLinear(
            self.dim,
            self.q_out_features + (2 * self.kv_out_features),
            bias=False,
        )
        self.o_proj = RuntimeLinear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, args.norm_eps)

        cache_seq_len = self.max_seq_len if self.use_cache else 1
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                int(args.max_batch_size),
                int(cache_seq_len),
                2,
                self.num_key_value_heads,
                self.head_dim,
            ),
            persistent=False,
        )
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.rope_head_dim,
                self.max_seq_len,
                theta=self.rope_theta,
                original_seq_len=self.original_seq_len,
                rope_factor=self.rope_factor,
                beta_fast=self.beta_fast,
                beta_slow=self.beta_slow,
            ),
            persistent=False,
        )

    def rebuild_runtime_buffers(self, max_seq_len: int | None = None) -> None:
        required = self.max_seq_len if max_seq_len is None else max(int(max_seq_len), 1)
        if required != int(self.freqs_cis.size(0)):
            self.freqs_cis = precompute_freqs_cis(
                self.rope_head_dim,
                required,
                theta=self.rope_theta,
                original_seq_len=self.original_seq_len,
                rope_factor=self.rope_factor,
                beta_fast=self.beta_fast,
                beta_slow=self.beta_slow,
            ).to(device=self.freqs_cis.device)
        if required != int(self.kv_cache.size(1)):
            self.kv_cache = self.kv_cache.new_zeros(
                int(self.kv_cache.size(0)),
                required,
                2,
                self.num_key_value_heads,
                self.head_dim,
            )

    def ensure_batch_capacity(self, batch_size: int) -> None:
        required = int(batch_size)
        if required <= int(self.kv_cache.size(0)):
            return
        grown = self.kv_cache.new_zeros(
            required,
            int(self.kv_cache.size(1)),
            2,
            self.num_key_value_heads,
            self.head_dim,
        )
        grown[: int(self.kv_cache.size(0))].copy_(self.kv_cache)
        self.kv_cache = grown

    def ensure_sequence_capacity(self, max_seq_len: int) -> None:
        required = int(max_seq_len)
        if required <= int(self.kv_cache.size(1)):
            return
        self.rebuild_runtime_buffers(required)

    def _reshape_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_raw, k_raw, v_raw = self.qkv_proj(x).split(
            [self.q_out_features, self.kv_out_features, self.kv_out_features],
            dim=-1,
        )
        q = q_raw.view(int(x.size(0)), int(x.size(1)), self.n_heads, self.head_dim)
        k = k_raw.view(
            int(x.size(0)),
            int(x.size(1)),
            self.num_key_value_heads,
            self.head_dim,
        )
        v = v_raw.view(
            int(x.size(0)),
            int(x.size(1)),
            self.num_key_value_heads,
            self.head_dim,
        )
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def _apply_rope(self, x: torch.Tensor, freqs_cis_pos: torch.Tensor) -> torch.Tensor:
        return apply_rotary_emb(x, freqs_cis_pos)

    def _cache_kv(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen = int(k.size(0)), int(k.size(1))
        end_pos = int(start_pos) + int(seqlen)
        self.ensure_batch_capacity(bsz)
        self.ensure_sequence_capacity(end_pos)
        with torch.no_grad():
            self.kv_cache[:bsz, int(start_pos) : end_pos, 0].copy_(k.detach())
            self.kv_cache[:bsz, int(start_pos) : end_pos, 1].copy_(v.detach())
        cached_k = self.kv_cache[:bsz, :end_pos, 0]
        cached_v = self.kv_cache[:bsz, :end_pos, 1]
        return cached_k, cached_v

    def reset(self) -> None:
        self.kv_cache.zero_()

    def refresh_state_buffers(self) -> None:
        self.kv_cache = torch.zeros_like(self.kv_cache)

    @staticmethod
    def _cuda_capability(device: torch.device) -> tuple[int, int]:
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(int(index))
        return int(props.major), int(props.minor)

    def _sdpa_flash_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        if str(q.device.type) != "cuda":
            raise RuntimeError("flash attention requires CUDA tensors")
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        attn_bias = self._causal_bias(q=q, k=k, start_pos=int(start_pos))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=attn_bias,
                is_causal=(attn_bias is None),
                enable_gqa=(int(q_t.size(1)) != int(k_t.size(1))),
            )
        return out.transpose(1, 2)

    @staticmethod
    def _causal_bias(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int,
    ) -> CausalBias | None:
        if int(start_pos) == 0:
            return None
        return causal_lower_right(int(q.size(1)), int(k.size(1)))

    def _eager_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        attn_bias = self._causal_bias(q=q, k=k, start_pos=int(start_pos))
        out = scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_bias,
            is_causal=(attn_bias is None),
            enable_gqa=(int(q_t.size(1)) != int(k_t.size(1))),
        )
        return out.transpose(1, 2)

    def _flash_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        major, minor = self._cuda_capability(q.device)
        if major in {8, 12}:
            return self._sdpa_flash_attention(q=q, k=k, v=v, start_pos=int(start_pos))
        raise RuntimeError(
            "unsupported CUDA capability for flash attention training: "
            f"sm{major}{minor}"
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q, k, v = self._reshape_qkv(x)

        q = self._apply_rope(q, freqs_cis)
        k = self._apply_rope(k, freqs_cis)

        if bool(self.use_cache):
            k_full, v_full = self._cache_kv(k=k, v=v, start_pos=int(start_pos))
        else:
            k_full, v_full = k, v

        if str(q.device.type) == "cuda":
            out = self._flash_attention(
                q=q,
                k=k_full,
                v=v_full,
                start_pos=int(start_pos),
            )
        else:
            out = self._eager_attention(
                q=q,
                k=k_full,
                v=v_full,
                start_pos=int(start_pos),
            )
        out = out.contiguous().view(bsz, seqlen, self.n_heads * self.head_dim)
        return self.o_proj(out)


__all__ = ["Attention"]
