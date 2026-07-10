from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from ml.training.posttrain.defaults import RELEASE_SFT_CURRICULUM_STAGE

_DEFAULT_POSTTRAIN_MAX_SEQ_LEN = 4096
_CURRICULUM_KEYS: dict[str, str] = {
    "sft": "sft_curriculum",
}


@dataclass(frozen=True)
class PosttrainStageSpec:
    stage_kind: str
    stage: str
    max_seq_len: int
    selection: str = "all_examples"
    crop_policy: str = "tail_tokens"
    recommended_share_of_updates: float = 1.0
    data: str = ""
    purpose: str = ""


def _missing_curriculum_message(resolved: Path) -> str:
    return (
        f"missing posttrain curriculum file: {resolved}. "
        "Build it first with "
        f"`ml-tool posttrain curriculum --sft_dir dataset/sft --output {resolved}`"
    )


def _load_curriculum(path: str) -> dict[str, object]:
    resolved = Path(os.path.abspath(str(path or "").strip()))
    if not resolved.is_file():
        raise FileNotFoundError(_missing_curriculum_message(resolved))
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"posttrain curriculum must be a JSON object: {resolved}")
    return payload


def resolve_posttrain_stage_spec(
    *,
    stage_kind: str,
    curriculum_path: str,
    curriculum_stage: str,
) -> PosttrainStageSpec | None:
    requested_stage = str(curriculum_stage or "").strip()
    if not requested_stage:
        return None
    key = _CURRICULUM_KEYS.get(str(stage_kind or "").strip().lower())
    if key is None:
        raise ValueError(f"unsupported posttrain stage kind: {stage_kind!r}")
    payload = _load_curriculum(curriculum_path)
    raw_stages = payload.get(key)
    if not isinstance(raw_stages, list):
        raise RuntimeError(f"curriculum key must be a list: {key!r}")
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            continue
        if str(raw_stage.get("stage") or "").strip() != requested_stage:
            continue
        max_seq_len = int(raw_stage.get("max_seq_len", 0) or 0)
        if max_seq_len <= 0:
            raise RuntimeError(
                f"curriculum stage is missing a valid max_seq_len: {requested_stage!r}"
            )
        return PosttrainStageSpec(
            stage_kind=str(stage_kind or "").strip().lower(),
            stage=requested_stage,
            max_seq_len=int(max_seq_len),
            selection=str(raw_stage.get("selection") or "all_examples"),
            crop_policy=str(raw_stage.get("crop_policy") or "tail_tokens"),
            recommended_share_of_updates=float(
                raw_stage.get("recommended_share_of_updates", 1.0) or 1.0
            ),
            data=str(raw_stage.get("data") or ""),
            purpose=str(raw_stage.get("purpose") or ""),
        )
    raise RuntimeError(
        "posttrain curriculum stage not found: "
        f"{requested_stage!r} under key {key!r}"
    )


def resolve_posttrain_max_seq_len(
    *,
    stage_kind: str,
    requested_max_seq_len: int,
    curriculum_path: str,
    curriculum_stage: str,
) -> int:
    stage = resolve_posttrain_stage_spec(
        stage_kind=stage_kind,
        curriculum_path=curriculum_path,
        curriculum_stage=curriculum_stage,
    )
    if stage is not None:
        return int(stage.max_seq_len)
    requested = int(requested_max_seq_len)
    if requested > 0:
        return requested
    return int(_DEFAULT_POSTTRAIN_MAX_SEQ_LEN)


def default_posttrain_curriculum_stage(*, stage_kind: str) -> str:
    normalized = str(stage_kind or "").strip().lower()
    if normalized == "sft":
        return str(RELEASE_SFT_CURRICULUM_STAGE)
    raise ValueError(f"unsupported posttrain stage kind: {stage_kind!r}")


__all__ = [
    "PosttrainStageSpec",
    "default_posttrain_curriculum_stage",
    "resolve_posttrain_max_seq_len",
    "resolve_posttrain_stage_spec",
]
