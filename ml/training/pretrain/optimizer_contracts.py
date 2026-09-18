from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

# Keep these aligned with PyTorch Muon defaults.
MUON_EPS = 1e-7
MUON_MOMENTUM = 0.95
MUON_NESTEROV = True
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_NS_STEPS = 5


@dataclass(frozen=True)
class OptimizerParamGroupSpec:
    name: str
    params: list[Tensor]
    lr: float | None = None
    weight_decay: float | None = None


@dataclass(frozen=True)
class TorchMuonHybridState:
    muon: dict[str, object]
    # fp32 master weights (one tensor per trainable parameter, in optimizer order).
    masters: list[Tensor]
    adamw: dict[str, object] | None = None
    kind: str = "torch_muon_hybrid"

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": str(self.kind),
            "muon": dict(self.muon),
            "adamw": None if self.adamw is None else dict(self.adamw),
            "masters": list(self.masters),
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
    ) -> TorchMuonHybridState:
        if str(payload.get("kind", "")) != "torch_muon_hybrid":
            raise RuntimeError(
                "Incompatible optimizer state for torch_muon_hybrid checkpoint."
            )
        muon_state = payload.get("muon")
        if not isinstance(muon_state, dict):
            raise RuntimeError("Missing Muon state in torch_muon_hybrid checkpoint.")
        adamw_state = payload.get("adamw")
        if adamw_state is not None and not isinstance(adamw_state, dict):
            raise RuntimeError("Invalid AdamW state in torch_muon_hybrid checkpoint.")
        masters = payload.get("masters")
        if not isinstance(masters, (list, tuple)):
            raise RuntimeError("Missing master weights in torch_muon_hybrid checkpoint.")
        if any(not isinstance(item, Tensor) for item in masters):
            raise RuntimeError("Invalid master weights in torch_muon_hybrid checkpoint.")
        return cls(
            muon=dict(muon_state),
            adamw=(None if adamw_state is None else dict(adamw_state)),
            masters=list(masters),
        )


__all__ = [
    "MUON_EPS",
    "MUON_MOMENTUM",
    "MUON_NESTEROV",
    "MUON_NS_COEFFICIENTS",
    "MUON_NS_STEPS",
    "OptimizerParamGroupSpec",
    "TorchMuonHybridState",
]
