from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from ml.training.posttrain.curriculum import resolve_posttrain_max_seq_len
from ml.training.posttrain.defaults import (
    RELEASE_SFT_EVAL_INTERVAL,
    RELEASE_SFT_LEARNING_RATE,
    RELEASE_SFT_SAVE_INTERVAL,
    RELEASE_SFT_TARGET_EXAMPLES_PER_UPDATE,
    RELEASE_SFT_WEIGHT_DECAY,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ml.training.posttrain.types import PosttrainStageArgs


def _max_seq_len_source(*, max_seq_len: int, curriculum_stage: str) -> str:
    if int(max_seq_len) > 0:
        return "explicit_arg"
    if str(curriculum_stage or "").strip():
        return f"curriculum:{str(curriculum_stage).strip()}"
    return "curriculum"


def _machine_recipe_int(
    recipe_payload: Mapping[str, object] | None,
    field: str,
    default: int,
) -> int:
    if recipe_payload is None:
        return int(default)
    if field not in recipe_payload:
        raise ValueError(f"machine recipe is missing required field {field!r}")
    raw = recipe_payload[field]
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"machine recipe field {field!r} must be an integer") from exc


@dataclass(frozen=True)
class SftReleaseDefaults:
    target_examples_per_update: int
    learning_rate: float
    weight_decay: float
    eval_interval: int
    save_interval: int


@dataclass(frozen=True)
class SftReleaseSemantics:
    curriculum_stage: str
    target_examples_per_update: int
    learning_rate: float
    weight_decay: float
    eval_interval: int
    save_interval: int
    resolved_max_seq_len: int
    max_seq_len_source: str
    name: str = ""
    notes: str = ""

    def to_payload(self, *, include_metadata: bool = False) -> dict[str, object]:
        payload = asdict(self)
        if not include_metadata:
            payload.pop("name", None)
            payload.pop("notes", None)
        return payload


RELEASE_SFT_DEFAULTS = SftReleaseDefaults(
    target_examples_per_update=RELEASE_SFT_TARGET_EXAMPLES_PER_UPDATE,
    learning_rate=RELEASE_SFT_LEARNING_RATE,
    weight_decay=RELEASE_SFT_WEIGHT_DECAY,
    eval_interval=RELEASE_SFT_EVAL_INTERVAL,
    save_interval=RELEASE_SFT_SAVE_INTERVAL,
)


def build_fixed_sft_release_semantics(
    *,
    curriculum_path: str,
    curriculum_stage: str,
    learning_rate: float,
    weight_decay: float,
    name: str = "release_sft",
    notes: str = "",
) -> SftReleaseSemantics:
    target = max(int(RELEASE_SFT_DEFAULTS.target_examples_per_update), 1)
    resolved_max_seq_len = resolve_posttrain_max_seq_len(
        stage_kind="sft",
        requested_max_seq_len=0,
        curriculum_path=str(curriculum_path),
        curriculum_stage=str(curriculum_stage),
    )
    return SftReleaseSemantics(
        name=str(name),
        curriculum_stage=str(curriculum_stage),
        target_examples_per_update=int(target),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        eval_interval=int(RELEASE_SFT_DEFAULTS.eval_interval),
        save_interval=int(RELEASE_SFT_DEFAULTS.save_interval),
        resolved_max_seq_len=int(resolved_max_seq_len),
        max_seq_len_source=f"curriculum:{str(curriculum_stage)}",
        notes=str(notes),
    )


def build_runtime_sft_release_semantics(
    *,
    args: PosttrainStageArgs,
    resolved_max_seq_len: int,
    machine_recipe: Mapping[str, object] | None = None,
) -> SftReleaseSemantics:
    batch_size = _machine_recipe_int(machine_recipe, "batch_size", int(args.batch_size))
    accumulation_steps = _machine_recipe_int(
        machine_recipe,
        "accumulation_steps",
        int(args.effective_accumulation_steps),
    )
    return SftReleaseSemantics(
        curriculum_stage=str(args.curriculum_stage or ""),
        target_examples_per_update=int(batch_size) * max(int(accumulation_steps), 1),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        eval_interval=int(args.eval_interval),
        save_interval=int(args.save_interval),
        resolved_max_seq_len=int(resolved_max_seq_len),
        max_seq_len_source=_max_seq_len_source(
            max_seq_len=int(args.max_seq_len),
            curriculum_stage=str(args.curriculum_stage or ""),
        ),
    )


__all__ = [
    "RELEASE_SFT_DEFAULTS",
    "SftReleaseDefaults",
    "SftReleaseSemantics",
    "build_fixed_sft_release_semantics",
    "build_runtime_sft_release_semantics",
]
