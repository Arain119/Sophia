from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as functional


_CAUSAL_CONV_BINARY_SHA256 = (
    "8309d50227c3ee8170225d1616ade91e5c031574e836d0c79f87ac621f0536b9"
)
_FLASH_KDA_TRAIN_BINARY_SHA256 = (
    "74d5cf52f3d2017f0449d5f6035dcdeec4dbae849f08ac4a1dc210062d397151"
)
_FFN_GRADIENT_SHAPES = (
    ((4096, 7936), "dz"),
    ((7936, 1536), "gate_weight"),
    ((4096, 1536), "x"),
    ((4096, 1536), "dy"),
    ((4096, 3968), "activation"),
)


@dataclass
class _DeviceRuntime:
    gate_stream: torch.cuda.Stream
    down_stream: torch.cuda.Stream


_RUNTIMES: dict[int, _DeviceRuntime] = {}


def _device_runtime(device: torch.device) -> _DeviceRuntime:
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    runtime = _RUNTIMES.get(index)
    if runtime is None:
        runtime = _DeviceRuntime(
            gate_stream=torch.cuda.Stream(device=torch.device("cuda", index)),
            down_stream=torch.cuda.Stream(device=torch.device("cuda", index)),
        )
        _RUNTIMES[index] = runtime
    return runtime


def _validate_ffn_gradient_inputs(inputs: tuple[torch.Tensor, ...]) -> None:
    device = inputs[0].device
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("concurrent FFN gradients require SM120")
    for value, (shape, name) in zip(inputs, _FFN_GRADIENT_SHAPES, strict=True):
        if value.device.type != "cuda" or value.dtype != torch.bfloat16:
            raise RuntimeError(f"{name} must be a CUDA BF16 tensor")
        if value.device != device:
            raise RuntimeError("concurrent FFN gradient tensors must share one device")
        if tuple(value.shape) != shape or not value.is_contiguous():
            raise RuntimeError(f"unsupported {name} shape or layout")


@torch.library.custom_op(
    "sophia::concurrent_ffn_grads_sm120",
    mutates_args=(),
    device_types="cuda",
)
def concurrent_ffn_grads_sm120(
    dz: torch.Tensor,
    gate_weight: torch.Tensor,
    x: torch.Tensor,
    dy: torch.Tensor,
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_ffn_gradient_inputs((dz, gate_weight, x, dy, activation))
    runtime = _device_runtime(dz.device)
    current = torch.cuda.current_stream(dz.device)
    runtime.gate_stream.wait_stream(current)
    runtime.down_stream.wait_stream(current)
    gate_grad = torch.empty_like(gate_weight)
    down_grad = dy.new_empty((dy.shape[-1], activation.shape[-1]))
    with torch.cuda.stream(runtime.gate_stream):
        torch.mm(dz.transpose(0, 1), x, out=gate_grad)
    with torch.cuda.stream(runtime.down_stream):
        torch.mm(dy.transpose(0, 1), activation, out=down_grad)
    dx = torch.empty_like(x)
    torch.mm(dz, gate_weight, out=dx)
    current.wait_stream(runtime.gate_stream)
    current.wait_stream(runtime.down_stream)
    return dx, gate_grad, down_grad


@concurrent_ffn_grads_sm120.register_fake
def _concurrent_ffn_grads_sm120_fake(
    dz: torch.Tensor,
    gate_weight: torch.Tensor,
    x: torch.Tensor,
    dy: torch.Tensor,
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        x.new_empty(x.shape),
        gate_weight.new_empty(gate_weight.shape),
        dy.new_empty((dy.shape[-1], activation.shape[-1])),
    )


class _ConcurrentFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, down_weight, gate_cap, up_cap):
        z = functional.linear(x, gate_weight)
        hidden = gate_weight.shape[0] // 2
        gate, up = z.split(hidden, dim=-1)
        gate_tanh = torch.tanh(gate / float(gate_cap))
        up_tanh = torch.tanh(up / float(up_cap))
        bounded_gate = float(gate_cap) * gate_tanh
        bounded_up = float(up_cap) * up_tanh
        activation = bounded_gate * torch.sigmoid(gate) * bounded_up
        output = functional.linear(activation, down_weight)
        ctx.save_for_backward(x, z, gate_weight, down_weight)
        ctx.gate_cap = float(gate_cap)
        ctx.up_cap = float(up_cap)
        return output

    @staticmethod
    def backward(ctx, dy):
        x, z, gate_weight, down_weight = ctx.saved_tensors
        hidden = z.shape[-1] // 2
        gate, up = z.split(hidden, dim=-1)
        gate_tanh = torch.tanh(gate / ctx.gate_cap)
        up_tanh = torch.tanh(up / ctx.up_cap)
        bounded_gate = ctx.gate_cap * gate_tanh
        bounded_up = ctx.up_cap * up_tanh
        sigmoid_gate = torch.sigmoid(gate)
        activation = bounded_gate * sigmoid_gate * bounded_up
        da = functional.linear(dy, down_weight.transpose(0, 1))
        d_gate = (
            da
            * bounded_up
            * (
                (1.0 - gate_tanh.square()) * sigmoid_gate
                + bounded_gate * sigmoid_gate * (1.0 - sigmoid_gate)
            )
        )
        d_up = da * bounded_gate * sigmoid_gate * (1.0 - up_tanh.square())
        dz = torch.cat((d_gate, d_up), dim=-1).to(dtype=dy.dtype)
        dx, gate_grad, down_grad = concurrent_ffn_grads_sm120(
            dz.reshape(-1, dz.shape[-1]),
            gate_weight,
            x.reshape(-1, x.shape[-1]),
            dy.reshape(-1, dy.shape[-1]),
            activation.reshape(-1, activation.shape[-1]),
        )
        return dx.reshape_as(x), gate_grad, down_grad, None, None


def _concurrent_ffn_forward(self, x):
    return _ConcurrentFFNFunction.apply(
        x,
        self.gate_up_proj.weight,
        self.down_proj.weight,
        self.gate_softcap,
        self.up_softcap,
    )


def enable_concurrent_ffn(model: torch.nn.Module) -> None:
    from ml.runtime.model.blocks import FeedForward

    modules = [module for module in model.modules() if isinstance(module, FeedForward)]
    if len(modules) != 28:
        raise RuntimeError(f"SM120 graph requires 28 FFNs, found {len(modules)}")
    for module in modules:
        gate_weight = module.gate_up_proj.weight
        down_weight = module.down_proj.weight
        if tuple(gate_weight.shape) != (7936, 1536):
            raise RuntimeError("SM120 graph found an unsupported gate weight")
        if tuple(down_weight.shape) != (1536, 3968):
            raise RuntimeError("SM120 graph found an unsupported down weight")
        if (
            gate_weight.device.type != "cuda"
            or down_weight.device != gate_weight.device
            or gate_weight.dtype != torch.bfloat16
            or down_weight.dtype != torch.bfloat16
        ):
            raise RuntimeError("SM120 graph requires CUDA BF16 FFN weights")
        module.forward = MethodType(_concurrent_ffn_forward, module)


def _install_attention_runtime(model: torch.nn.Module) -> None:
    from causal_conv1d import causal_conv1d_fn
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.common.gate import fused_beta_sigmoid, fused_beta_sigmoid_bwd
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
    from flash_kda.train.pipeline import chunk_kda_train_bwd, chunk_kda_train_fwd
    from ml.runtime.model.attention import CausalDepthwiseConv1d, SophiaKDA

    reference_conv = CausalDepthwiseConv1d.forward

    def causal_conv(self, x, *, history=None):
        if history is not None:
            return reference_conv(self, x, history=history)
        output = causal_conv1d_fn(
            x.transpose(1, 2),
            self.weight.to(dtype=x.dtype),
            activation="silu",
        )
        return output.transpose(1, 2), None

    class FlashKDAFunction(torch.autograd.Function):
        @staticmethod
        @input_guard
        @autocast_custom_fwd
        def forward(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            scale,
            initial_state,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            state_v_first=False,
            cu_seqlens=None,
            cu_seqlens_cpu=None,
            safe_gate=False,
            lower_bound=None,
            chunk_size=64,
        ):
            q_rstd, k_rstd = None, None
            if use_qk_l2norm_in_kernel:
                q, q_rstd = l2norm_fwd(q)
                k, k_rstd = l2norm_fwd(k)
            beta_raw = beta
            if use_beta_sigmoid_in_kernel:
                beta = fused_beta_sigmoid(
                    beta_raw, scale=2.0 if allow_neg_eigval else 1.0
                )
            chunk_indices = None
            if cu_seqlens is not None:
                from flash_kda.train import prepare_chunk_indices

                chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            output, final_state, g_cumsum, aqk, akk = chunk_kda_train_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=a_log,
                dt_bias=dt_bias,
                chunk_size=chunk_size,
                state_v_first=state_v_first,
            )
            ctx.save_for_backward(
                q,
                q_rstd,
                k,
                k_rstd,
                v,
                g_cumsum,
                g,
                beta_raw,
                beta,
                a_log,
                dt_bias,
                aqk,
                akk,
                initial_state,
                cu_seqlens,
                chunk_indices,
            )
            ctx.chunk_size = int(chunk_size)
            ctx.safe_gate = bool(safe_gate)
            ctx.scale = float(scale)
            ctx.lower_bound = lower_bound
            ctx.use_qk_l2norm_in_kernel = bool(use_qk_l2norm_in_kernel)
            ctx.use_gate_in_kernel = bool(use_gate_in_kernel)
            ctx.use_beta_sigmoid_in_kernel = bool(use_beta_sigmoid_in_kernel)
            ctx.allow_neg_eigval = bool(allow_neg_eigval)
            ctx.state_v_first = bool(state_v_first)
            return output.type_as(q), final_state

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(ctx, do, dht):
            (
                q,
                q_rstd,
                k,
                k_rstd,
                v,
                g_cumsum,
                g_input,
                beta_raw,
                beta,
                a_log,
                dt_bias,
                aqk,
                akk,
                initial_state,
                cu_seqlens,
                chunk_indices,
            ) = ctx.saved_tensors
            dq, dk, dv, db, dg, dh0, d_a, dbias = chunk_kda_train_bwd(
                q=q,
                k=k,
                v=v,
                beta=beta,
                Aqk=aqk,
                Akk=akk,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                g=g_cumsum,
                g_org=g_input if ctx.use_gate_in_kernel else None,
                state_v_first=ctx.state_v_first,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=ctx.chunk_size,
                safe_gate=ctx.safe_gate,
                lower_bound=ctx.lower_bound,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                A_log=a_log,
                dt_bias=dt_bias,
            )
            if ctx.use_qk_l2norm_in_kernel:
                dq = l2norm_bwd(q, q_rstd, dq)
                dk = l2norm_bwd(k, k_rstd, dk)
            if ctx.use_beta_sigmoid_in_kernel:
                db = fused_beta_sigmoid_bwd(
                    beta_raw,
                    db,
                    scale=2.0 if ctx.allow_neg_eigval else 1.0,
                )
            return (
                dq.to(q),
                dk.to(k),
                dv.to(v),
                dg.to(g_input),
                db.to(beta_raw),
                d_a,
                dbias,
                None,
                dh0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    def flash_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log,  # noqa: N803
        dt_bias,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        safe_gate=False,
        lower_bound=None,
        chunk_size=64,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(sorted(kwargs))
        if scale is None:
            scale = q.shape[-1] ** -0.5
        return FlashKDAFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            A_log.float().contiguous(),
            dt_bias.float().contiguous(),
            float(scale),
            initial_state,
            bool(output_final_state),
            bool(use_qk_l2norm_in_kernel),
            bool(use_gate_in_kernel),
            bool(use_beta_sigmoid_in_kernel),
            bool(allow_neg_eigval),
            bool(state_v_first),
            cu_seqlens,
            cu_seqlens_cpu,
            bool(safe_gate),
            lower_bound,
            int(chunk_size),
        )

    compiled_kda = torch.compiler.disable(flash_kda)
    for module in model.modules():
        if isinstance(module, CausalDepthwiseConv1d):
            module.forward = MethodType(causal_conv, module)
        elif isinstance(module, SophiaKDA):
            module._fla_chunk_kda = lambda: compiled_kda


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_sm120_runtime(model: torch.nn.Module) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 graph requires CUDA")
    device = next(model.parameters()).device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("SM120 graph requires CUDA capability 12.0")
    if str(torch.__version__) != "2.8.0+cu128" or str(torch.version.cuda) != "12.8":
        raise RuntimeError("SM120 graph requires PyTorch 2.8.0+cu128")
    try:
        causal_binary = import_module("causal_conv1d_cuda")
        flash_binary = import_module("flash_kda_train_C")
        import_module("flash_kda.train.pipeline")
        fla = import_module("fla")
        triton = import_module("triton")
    except ImportError as exc:
        raise RuntimeError(
            "SM120 graph requires the qualified causal-conv and FlashKDA extensions"
        ) from exc
    if str(getattr(fla, "__version__", "")) != "0.5.2":
        raise RuntimeError("SM120 graph requires flash-linear-attention 0.5.2")
    if str(getattr(triton, "__version__", "")) != "3.4.0":
        raise RuntimeError("SM120 graph requires Triton 3.4.0")
    causal_hash = _sha256(str(causal_binary.__file__))
    flash_hash = _sha256(str(flash_binary.__file__))
    if causal_hash != _CAUSAL_CONV_BINARY_SHA256:
        raise RuntimeError("SM120 causal-conv binary hash mismatch")
    if flash_hash != _FLASH_KDA_TRAIN_BINARY_SHA256:
        raise RuntimeError("SM120 FlashKDA training binary hash mismatch")
    _install_attention_runtime(model)
    _install_mla_telemetry_runtime(model)
    enable_concurrent_ffn(model)
    return {
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "triton_version": str(triton.__version__),
        "fla_version": str(fla.__version__),
        "causal_conv_binary_sha256": causal_hash,
        "flash_kda_train_binary_sha256": flash_hash,
        "mla_logit_telemetry_backend": "triton_causal_max_sm120",
    }


def _install_mla_telemetry_runtime(model: torch.nn.Module) -> None:
    from ml.runtime.model.attention import SophiaMLA
    from ml.training.pretrain.engine.sm120_mla_telemetry import (
        record_mla_attention_logit_max_sm120,
    )

    modules = [module for module in model.modules() if isinstance(module, SophiaMLA)]
    if len(modules) != 7:
        raise RuntimeError(f"SM120 graph requires 7 MLA modules, found {len(modules)}")
    for module in modules:
        module._record_attention_logit_max = MethodType(
            record_mla_attention_logit_max_sm120,
            module,
        )


__all__ = [
    "concurrent_ffn_grads_sm120",
    "install_sm120_runtime",
]
