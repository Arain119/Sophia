from __future__ import annotations

from ml.training.pretrain.value_semantics import normalized_str

SHORT_PRETRAIN_RUN_KINDS = frozenset(
    {
        "pretrain_machine_validate",
        "pretrain_stability_probe",
        "pretrain_check",
    }
)
def is_budgeted_short_pretrain_run(value: object) -> bool:
    return normalized_str(value, default="pretrain") in SHORT_PRETRAIN_RUN_KINDS


__all__ = [
    "SHORT_PRETRAIN_RUN_KINDS",
    "is_budgeted_short_pretrain_run",
]
