from __future__ import annotations

from importlib import import_module
import math

import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.attention.bias import CausalBias, causal_lower_right

from ml.runtime.model.config import ModelArgs
from ml.runtime.model.ops import RMSNorm
from ml.runtime.model.runtime_linear import RuntimeLinear
from ml.runtime.model.state import LayerCacheSnapshot


PAD_LOGIT = -1.0e4


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(self.channels, self.kernel_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        *,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C], history: [B, C, K-1]
        x_channels = x.transpose(1, 2)
        history_length = self.kernel_size - 1
        if history is None and history_length > 0:
            history = x_channels.new_zeros(
                int(x.size(0)), self.channels, history_length
            )
        sequence = (
            x_channels
            if history_length == 0
            else torch.cat((history.to(dtype=x.dtype), x_channels), dim=-1)
        )
        output = functional.conv1d(
            sequence,
            self.weight.to(dtype=x.dtype).unsqueeze(1),
            groups=self.channels,
        ).transpose(1, 2)
        next_history = (
            sequence[..., :0]
            if history_length == 0
            else sequence[..., -history_length:]
        )
        return functional.silu(output), next_history


class SophiaKDA(nn.Module):
    """Kimi Delta Attention adapted as Sophia's recurrent token mixer."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.dim = int(args.dim)
        self.num_heads = int(args.num_heads)
        self.head_dim = int(args.head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.decay_rank = int(args.kda_decay_rank)
        self.output_gate_rank = int(args.kda_output_gate_rank)
        self.output_gate_full_rank = bool(args.kda_output_gate_full_rank)
        self.lower_bound = float(args.kda_decay_lower_bound)
        self.dt_min = float(args.kda_dt_min)
        self.dt_max = float(args.kda_dt_max)
        self.dt_floor = float(args.kda_dt_floor)
        self.backend = str(args.kda_backend)
        self.use_cache = bool(args.use_cache)
        self.conv_kernel = int(args.short_conv_kernel)

        self.q_proj = RuntimeLinear(self.dim, self.inner_dim, bias=False)
        self.k_proj = RuntimeLinear(self.dim, self.inner_dim, bias=False)
        self.v_proj = RuntimeLinear(self.dim, self.inner_dim, bias=False)

        self.q_conv = CausalDepthwiseConv1d(self.inner_dim, self.conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(self.inner_dim, self.conv_kernel)
        self.v_conv = CausalDepthwiseConv1d(self.inner_dim, self.conv_kernel)

        self.decay_down = RuntimeLinear(self.dim, self.decay_rank, bias=False)
        self.decay_up = RuntimeLinear(self.decay_rank, self.inner_dim, bias=False)
        self.beta_proj = RuntimeLinear(self.dim, self.num_heads, bias=False)
        self.A_log = nn.Parameter(
            torch.full(
                (self.num_heads,),
                float(args.kda_a_log_init),
                dtype=torch.float32,
            )
        )
        dt = torch.exp(
            torch.rand(self.inner_dim, dtype=torch.float32)
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp_min(self.dt_floor)
        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        if self.output_gate_full_rank:
            self.output_gate = RuntimeLinear(self.dim, self.inner_dim, bias=False)
        else:
            self.output_gate_down = RuntimeLinear(
                self.dim, self.output_gate_rank, bias=False
            )
            self.output_gate_up = RuntimeLinear(
                self.output_gate_rank, self.inner_dim, bias=False
            )
        self.output_norm = RMSNorm(self.head_dim, args.norm_eps)
        self.o_proj = RuntimeLinear(self.inner_dim, self.dim, bias=False)

        self.register_buffer(
            "recurrent_state",
            torch.zeros(
                int(args.max_batch_size),
                self.num_heads,
                self.head_dim,
                self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "conv_state",
            torch.zeros(
                int(args.max_batch_size),
                3,
                self.inner_dim,
                self.conv_kernel - 1,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def ensure_batch_capacity(self, batch_size: int) -> None:
        required = int(batch_size)
        if required <= int(self.recurrent_state.size(0)):
            return
        recurrent = self.recurrent_state.new_zeros(
            required, self.num_heads, self.head_dim, self.head_dim
        )
        recurrent[: self.recurrent_state.size(0)].copy_(self.recurrent_state)
        conv = self.conv_state.new_zeros(
            required, 3, self.inner_dim, self.conv_kernel - 1
        )
        conv[: self.conv_state.size(0)].copy_(self.conv_state)
        self.recurrent_state = recurrent
        self.conv_state = conv

    def ensure_sequence_capacity(self, max_seq_len: int) -> None:
        del max_seq_len

    def rebuild_runtime_buffers(self, max_seq_len: int) -> None:
        del max_seq_len
        device = self.q_proj.weight.device
        self.recurrent_state = torch.zeros(
            int(self.recurrent_state.size(0)),
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=torch.float32,
        )
        self.conv_state = torch.zeros(
            int(self.conv_state.size(0)),
            3,
            self.inner_dim,
            self.conv_kernel - 1,
            device=device,
            dtype=torch.float32,
        )

    def reset(self) -> None:
        if self.recurrent_state.is_inference() or self.conv_state.is_inference():
            self.recurrent_state = torch.zeros_like(self.recurrent_state)
            self.conv_state = torch.zeros_like(self.conv_state)
        else:
            self.recurrent_state.zero_()
            self.conv_state.zero_()

    def refresh_state_buffers(self) -> None:
        self.rebuild_runtime_buffers(0)

    def cache_snapshot(
        self, *, device: str, batch_size: int, cache_pos: int | None
    ) -> LayerCacheSnapshot:
        del cache_pos
        return LayerCacheSnapshot(
            recurrent=self.recurrent_state[:batch_size].to(device).clone(),
            conv=self.conv_state[:batch_size].to(device).clone(),
        )

    def validate_cache_snapshot(self, snapshot: LayerCacheSnapshot) -> None:
        if snapshot.latent is not None:
            raise ValueError("KDA cache cannot contain MLA latent state")
        expected_recurrent = tuple(self.recurrent_state.shape[1:])
        expected_conv = tuple(self.conv_state.shape[1:])
        if snapshot.recurrent is not None and tuple(snapshot.recurrent.shape[1:]) != expected_recurrent:
            raise ValueError("KDA recurrent cache shape mismatch")
        if snapshot.conv is not None and tuple(snapshot.conv.shape[1:]) != expected_conv:
            raise ValueError("KDA convolution cache shape mismatch")

    def load_cache_snapshot(self, snapshot: LayerCacheSnapshot) -> None:
        self.validate_cache_snapshot(snapshot)
        required = snapshot.batch_size()
        self.ensure_batch_capacity(required)
        if snapshot.recurrent is not None:
            self.recurrent_state[:required].copy_(
                snapshot.recurrent.to(self.recurrent_state.device, dtype=torch.float32)
            )
        if snapshot.conv is not None:
            self.conv_state[:required].copy_(
                snapshot.conv.to(self.conv_state.device, dtype=torch.float32)
            )

    def _short_convolution(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = int(q.size(0))
        histories: list[torch.Tensor | None]
        if self.use_cache and int(start_pos) > 0:
            histories = [self.conv_state[:bsz, index] for index in range(3)]
        else:
            histories = [None, None, None]
        outputs = []
        next_histories = []
        for module, value, history in zip(
            (self.q_conv, self.k_conv, self.v_conv),
            (q, k, v),
            histories,
            strict=True,
        ):
            output, next_history = module(value, history=history)
            outputs.append(output)
            next_histories.append(next_history)
        if self.use_cache:
            self.ensure_batch_capacity(bsz)
            with torch.no_grad():
                for index, history in enumerate(next_histories):
                    self.conv_state[:bsz, index].copy_(
                        history.detach().to(dtype=torch.float32)
                    )
        return outputs[0], outputs[1], outputs[2]

    def _reference_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_logits: torch.Tensor,
        beta_logits: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_dtype = v.dtype
        q = functional.normalize(q.float(), p=2.0, dim=-1) * (
            self.head_dim ** -0.5
        )
        k = functional.normalize(k.float(), p=2.0, dim=-1)
        v = v.float()
        scale = torch.exp(self.A_log).view(1, 1, self.num_heads, 1)
        bias = self.dt_bias.view(1, 1, self.num_heads, self.head_dim)
        log_decay = self.lower_bound * torch.sigmoid(
            scale * (decay_logits.float() + bias)
        )
        alpha = torch.exp(log_decay)
        beta = torch.sigmoid(beta_logits.float())
        state = (
            q.new_zeros(int(q.size(0)), self.num_heads, self.head_dim, self.head_dim)
            if initial_state is None
            else initial_state.float()
        )
        outputs: list[torch.Tensor] = []
        for index in range(int(q.size(1))):
            state = state * alpha[:, index].unsqueeze(-1)
            prediction = torch.einsum("bhd,bhdv->bhv", k[:, index], state)
            correction = (v[:, index] - prediction) * beta[:, index].unsqueeze(-1)
            state = state + k[:, index].unsqueeze(-1) * correction.unsqueeze(-2)
            outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, index], state))
        return torch.stack(outputs, dim=1).to(dtype=output_dtype), state

    @staticmethod
    def _fla_fused_recurrent_kda():
        try:
            return import_module("fla.ops.kda").fused_recurrent_kda
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "CUDA Sophia KDA decoding requires flash-linear-attention>=0.5.2"
            ) from exc

    @staticmethod
    def _fla_chunk_kda():
        try:
            return import_module("fla.ops.kda").chunk_kda
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "CUDA Sophia KDA training requires flash-linear-attention>=0.5.2"
            ) from exc

    def _run_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_logits: torch.Tensor,
        beta_logits: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_fla = q.device.type == "cuda" and self.backend != "reference"
        if self.backend == "fla" and q.device.type != "cuda":
            raise RuntimeError("kda_backend='fla' requires CUDA tensors")
        if not use_fla:
            return self._reference_kda(
                q, k, v, decay_logits, beta_logits, initial_state
            )
        if int(q.size(1)) == 1 and initial_state is not None:
            # Single-step decode: advancing the state by one position does not
            # need the chunk grid.
            output, final_state = self._fla_fused_recurrent_kda()(
                q=q,
                k=k,
                v=v,
                g=decay_logits,
                beta=beta_logits,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=self.lower_bound,
            )
            return output, final_state
        chunk_kda = self._fla_chunk_kda()
        output, final_state = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=decay_logits,
            beta=beta_logits,
            initial_state=initial_state,
            output_final_state=bool(self.use_cache),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=self.lower_bound,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )
        if final_state is None:
            final_state = q.new_zeros(
                int(q.size(0)), self.num_heads, self.head_dim, self.head_dim,
                dtype=torch.float32,
            )
        return output, final_state

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        if self.use_cache:
            self.ensure_batch_capacity(int(bsz))
        # Padding only ever exists in the prompt, so it applies to the prefill
        # alone; every decode step is a real token for every row.
        pad_mask = getattr(self, "_pad_mask", None) if int(start_pos) == 0 else None
        if pad_mask is not None:
            # Zero the input so the causal depthwise convolution sees exactly the
            # zeros an unpadded sequence sees in its own left padding.
            keep = pad_mask[:, :seqlen].to(dtype=x.dtype).unsqueeze(-1)
            x = x * keep
        q, k, v = self._short_convolution(
            self.q_proj(x), self.k_proj(x), self.v_proj(x), start_pos=int(start_pos)
        )
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim)
        decay_logits = self.decay_up(self.decay_down(x)).view_as(q)
        beta_logits = self.beta_proj(x)
        if pad_mask is not None:
            # alpha -> 1 and beta -> 0, so the state passes a pad through unchanged.
            drop = ~pad_mask[:, :seqlen].bool()
            decay_logits = decay_logits.masked_fill(drop[:, :, None, None], PAD_LOGIT)
            beta_logits = beta_logits.masked_fill(drop[:, :, None], PAD_LOGIT)
        initial_state = (
            self.recurrent_state[:bsz]
            if self.use_cache and int(start_pos) > 0
            else None
        )
        output, final_state = self._run_kda(
            q, k, v, decay_logits, beta_logits, initial_state
        )
        if self.use_cache:
            with torch.no_grad():
                self.recurrent_state[:bsz].copy_(
                    final_state.detach().to(dtype=torch.float32)
                )
        gate_logits = (
            self.output_gate(x)
            if self.output_gate_full_rank
            else self.output_gate_up(self.output_gate_down(x))
        )
        gate = torch.sigmoid(gate_logits).view(
            bsz, seqlen, self.num_heads, self.head_dim
        )
        output = self.output_norm(output.to(dtype=x.dtype)) * gate
        return self.o_proj(output.reshape(bsz, seqlen, self.inner_dim))


class SophiaMLA(nn.Module):
    """NoPE multi-head latent attention with a full-rank output gate."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.dim = int(args.dim)
        self.num_heads = int(args.num_heads)
        self.head_dim = int(args.head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.q_rank = int(args.mla_q_rank)
        self.kv_rank = int(args.mla_kv_rank)
        self.use_cache = bool(args.use_cache)
        self.logit_telemetry_enabled = False

        self.q_down = RuntimeLinear(self.dim, self.q_rank, bias=False)
        self.q_norm = RMSNorm(self.q_rank, args.norm_eps)
        self.q_up = RuntimeLinear(self.q_rank, self.inner_dim, bias=False)

        self.kv_down = RuntimeLinear(self.dim, self.kv_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_rank, args.norm_eps)
        self.k_up = RuntimeLinear(self.kv_rank, self.inner_dim, bias=False)
        self.v_up = RuntimeLinear(self.kv_rank, self.inner_dim, bias=False)

        self.output_gate = RuntimeLinear(self.dim, self.inner_dim, bias=False)
        self.o_proj = RuntimeLinear(self.inner_dim, self.dim, bias=False)
        self.register_buffer(
            "latent_cache",
            torch.zeros(
                int(args.max_batch_size), int(args.max_seq_len), self.kv_rank
            ),
            persistent=False,
        )
        self.register_buffer(
            "attention_logit_max",
            torch.full((self.num_heads,), float("-inf"), dtype=torch.float32),
            persistent=False,
        )

    def enable_attention_logit_telemetry(
        self,
        buffer: torch.Tensor | None = None,
    ) -> None:
        self.logit_telemetry_enabled = True
        if buffer is None:
            buffer = torch.full(
                (self.num_heads,),
                float("-inf"),
                device=self.q_up.weight.device,
                dtype=torch.float32,
            )
        if (
            tuple(buffer.shape) != (self.num_heads,)
            or buffer.device != self.q_up.weight.device
            or buffer.dtype != torch.float32
        ):
            raise ValueError("MLA attention-logit buffer must be FP32 [num_heads]")
        self.attention_logit_max = buffer

    def reset_attention_logit_max(self) -> None:
        self.attention_logit_max.fill_(float("-inf"))

    @torch.no_grad()
    def _record_attention_logit_max(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> None:
        query_tile = 512
        query_length = int(q.size(-2))
        key_length = int(k.size(-2))
        query_offset = key_length - query_length
        if query_offset < 0:
            raise ValueError("MLA telemetry requires key length >= query length")
        tile_causal = torch.ones(
            (min(query_tile, query_length), min(query_tile, query_length)),
            device=q.device,
            dtype=torch.bool,
        ).tril_()
        key_transposed = k.detach().transpose(-2, -1)
        running = self.attention_logit_max
        scale = self.head_dim**-0.5
        for query_start in range(0, query_length, query_tile):
            query_end = min(query_start + query_tile, query_length)
            key_end = query_offset + query_end
            tile_width = query_end - query_start
            logits = torch.matmul(
                q.detach()[..., query_start:query_end, :],
                key_transposed[..., :key_end],
            ) * float(scale)
            logits[..., -tile_width:].masked_fill_(
                ~tile_causal[:tile_width, :tile_width].view(
                    1, 1, tile_width, tile_width
                ),
                float("-inf"),
            )
            running = torch.maximum(
                running,
                logits.amax(dim=(0, 2, 3)).float(),
            )
        self.attention_logit_max.copy_(running)

    def ensure_batch_capacity(self, batch_size: int) -> None:
        required = int(batch_size)
        if required <= int(self.latent_cache.size(0)):
            return
        grown = self.latent_cache.new_zeros(
            required, int(self.latent_cache.size(1)), self.kv_rank
        )
        grown[: self.latent_cache.size(0)].copy_(self.latent_cache)
        self.latent_cache = grown

    def ensure_sequence_capacity(self, max_seq_len: int) -> None:
        required = int(max_seq_len)
        if required <= int(self.latent_cache.size(1)):
            return
        grown = self.latent_cache.new_zeros(
            int(self.latent_cache.size(0)), required, self.kv_rank
        )
        grown[:, : self.latent_cache.size(1)].copy_(self.latent_cache)
        self.latent_cache = grown

    def rebuild_runtime_buffers(self, max_seq_len: int) -> None:
        self.latent_cache = self.kv_down.weight.new_zeros(
            int(self.latent_cache.size(0)), int(max_seq_len), self.kv_rank
        )

    def reset(self) -> None:
        if self.latent_cache.is_inference():
            self.latent_cache = torch.zeros_like(self.latent_cache)
        else:
            self.latent_cache.zero_()

    def refresh_state_buffers(self) -> None:
        self.latent_cache = torch.zeros_like(self.latent_cache)

    def cache_snapshot(
        self, *, device: str, batch_size: int, cache_pos: int | None
    ) -> LayerCacheSnapshot:
        end = int(self.latent_cache.size(1) if cache_pos is None else cache_pos)
        return LayerCacheSnapshot(
            latent=self.latent_cache[:batch_size, :end].to(device).clone()
        )

    def validate_cache_snapshot(self, snapshot: LayerCacheSnapshot) -> None:
        if snapshot.recurrent is not None or snapshot.conv is not None:
            raise ValueError("MLA cache cannot contain KDA state")
        if snapshot.latent is not None and int(snapshot.latent.size(-1)) != self.kv_rank:
            raise ValueError("MLA latent cache shape mismatch")

    def load_cache_snapshot(self, snapshot: LayerCacheSnapshot) -> None:
        self.validate_cache_snapshot(snapshot)
        if snapshot.latent is None:
            return
        batch, length = int(snapshot.latent.size(0)), int(snapshot.latent.size(1))
        self.ensure_batch_capacity(batch)
        self.ensure_sequence_capacity(length)
        self.latent_cache[:batch, :length].copy_(
            snapshot.latent.to(
                self.latent_cache.device, dtype=self.latent_cache.dtype
            )
        )

    def _resolve_key_pad(
        self, start_pos: int, bsz: int, k_len: int, device: torch.device
    ) -> torch.Tensor | None:
        """Which key positions are real, over the whole cached span."""
        if int(start_pos) == 0:
            mask = getattr(self, "_pad_mask", None)
            self._key_pad = None if mask is None else mask[:bsz].bool().to(device)
            return self._key_pad
        cached = getattr(self, "_key_pad", None)
        if cached is None:
            return None
        prompt_len = int(cached.size(1))
        if k_len <= prompt_len:
            return cached[:bsz, :k_len]
        grown = torch.ones((bsz, k_len), dtype=torch.bool, device=device)
        grown[:, :prompt_len] = cached[:bsz]
        return grown

    @staticmethod
    def _causal_bias(q: torch.Tensor, k: torch.Tensor, start_pos: int) -> CausalBias | None:
        if int(start_pos) == 0:
            return None
        return causal_lower_right(int(q.size(1)), int(k.size(1)))

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_up(self.q_norm(self.q_down(x))).view(
            bsz, seqlen, self.num_heads, self.head_dim
        )
        latent = self.kv_norm(self.kv_down(x))
        if self.use_cache:
            end = int(start_pos) + int(seqlen)
            self.ensure_batch_capacity(int(bsz))
            self.ensure_sequence_capacity(end)
            with torch.no_grad():
                self.latent_cache[:bsz, int(start_pos) : end].copy_(latent.detach())
            latent_full = self.latent_cache[:bsz, :end]
        else:
            latent_full = latent
        k = self.k_up(latent_full).view(
            bsz, int(latent_full.size(1)), self.num_heads, self.head_dim
        )
        v = self.v_up(latent_full).view_as(k)
        q_t, k_t, v_t = (value.transpose(1, 2) for value in (q, k, v))
        if self.logit_telemetry_enabled:
            self._record_attention_logit_max(q_t, k_t)
        key_pad = self._resolve_key_pad(int(start_pos), int(bsz), int(k.size(1)), x.device)
        if key_pad is None:
            bias = self._causal_bias(q, k, int(start_pos))
            output = functional.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=bias,
                is_causal=(bias is None),
            ).transpose(1, 2)
        else:
            # causal_lower_right is an opaque bias object, so when padding is in
            # play build the whole additive mask instead of trying to combine.
            q_len, k_len = int(q.size(1)), int(k.size(1))
            allow = key_pad[:, None, None, :].expand(bsz, 1, q_len, k_len).clone()
            if q_len > 1:
                offset = k_len - q_len
                causal = torch.ones((q_len, k_len), dtype=torch.bool, device=x.device).tril(offset)
                allow &= causal[None, None]
            attn_mask = torch.zeros((bsz, 1, q_len, k_len), dtype=q_t.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(~allow, float("-inf"))
            output = functional.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask, is_causal=False
            ).transpose(1, 2)
        gate = torch.sigmoid(self.output_gate(x)).view_as(output)
        gated_output = (output.float() * gate.float()).to(dtype=x.dtype)
        return self.o_proj(gated_output.reshape(bsz, seqlen, self.inner_dim))


def set_pad_mask(model: nn.Module, mask: torch.Tensor | None) -> None:
    """Hang a [B, T] boolean mask (True = real token) on every mixer, or clear it.

    Clearing also drops MLA's remembered key mask, so a later unpadded batch cannot
    inherit the previous batch's padding.
    """
    for module in model.modules():
        if isinstance(module, (SophiaKDA, SophiaMLA)):
            module._pad_mask = mask
            if mask is None and isinstance(module, SophiaMLA):
                module._key_pad = None


__all__ = [
    "CausalDepthwiseConv1d",
    "SophiaKDA",
    "SophiaMLA",
    "set_pad_mask",
]
