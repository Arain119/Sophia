from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn



def infer_module_tensor_device(
    module: nn.Module,
    *,
    default_device: torch.device,
) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        if isinstance(buffer, torch.Tensor):
            return buffer.device
    return default_device


def apply_preserving_complex_buffers(
    module: nn.Module,
    fn,
    *,
    buffer_names: tuple[str, ...],
    apply_super,
):
    preserved: dict[str, torch.Tensor] = {}
    preserved_device = torch.device("cpu")
    for name in buffer_names:
        tensor = module._buffers.get(name)
        if isinstance(tensor, torch.Tensor) and tensor.is_complex():
            preserved[name] = tensor
            preserved_device = tensor.device
            module._buffers[name] = None
    try:
        result = apply_super()
    finally:
        if preserved:
            target_device = infer_module_tensor_device(
                module,
                default_device=preserved_device,
            )
            for name, tensor in preserved.items():
                module._buffers[name] = tensor.to(device=target_device)
    return result

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.weight._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.rms_norm(
            x,
            (int(x.size(-1)),),
            self.weight,
            float(self.eps),
        )


class StandardLogitMixer(nn.Module):
    """Decoder logits path: final norm followed by output projection."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        norm: RMSNorm,
        output: nn.Module,
    ) -> torch.Tensor:
        return output(norm(x))
