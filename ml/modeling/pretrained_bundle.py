from __future__ import annotations

from dataclasses import fields, is_dataclass
from dataclasses import asdict
import json
import os
from typing import Protocol, TypeVar

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file


class SupportsSavePretrained(Protocol):
    def save_pretrained(self, save_directory: str | os.PathLike) -> None: ...


ConfigT = TypeVar("ConfigT")


def save_canonical_config_bundle(
    config: object,
    *,
    save_directory: str | os.PathLike,
) -> None:
    save_dir = os.path.abspath(str(save_directory))
    os.makedirs(save_dir, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else dict(vars(config))
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_pretrained_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    save_directory: str | os.PathLike,
    safe_serialization: bool,
) -> None:
    save_dir = os.path.abspath(str(save_directory))
    os.makedirs(save_dir, exist_ok=True)
    export_state = {
        str(key): value.detach().cpu()
        for key, value in dict(state_dict).items()
    }
    if bool(safe_serialization):
        save_safetensors_file(
            export_state,
            os.path.join(save_dir, "model.safetensors"),
        )
        bin_path = os.path.join(save_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            os.remove(bin_path)
        return
    torch.save(export_state, os.path.join(save_dir, "pytorch_model.bin"))


def load_pretrained_state_dict(
    export_dir: str | os.PathLike,
) -> dict[str, torch.Tensor] | None:
    resolved_export_dir = os.path.abspath(str(export_dir))
    safetensors_path = os.path.join(resolved_export_dir, "model.safetensors")
    pytorch_path = os.path.join(resolved_export_dir, "pytorch_model.bin")
    if os.path.isfile(safetensors_path):
        return load_safetensors_file(safetensors_path)
    if os.path.isfile(pytorch_path):
        raw_state = torch.load(pytorch_path, map_location="cpu", weights_only=True)
        if not isinstance(raw_state, dict):
            raise RuntimeError(
                f"unexpected state_dict payload type: {type(raw_state)!r}"
            )
        return raw_state
    return None


def load_canonical_config_from_pretrained(
    pretrained_model_name_or_path: str | os.PathLike,
    *,
    config_cls: type[ConfigT],
) -> ConfigT:
    config_path = os.path.join(
        os.path.abspath(str(pretrained_model_name_or_path)),
        "config.json",
    )
    try:
        with open(config_path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing exported config.json: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid exported config.json: {config_path}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"exported config.json must contain a JSON object: {config_path}"
        )
    from_mapping = getattr(config_cls, "from_mapping", None)
    if callable(from_mapping):
        canonical_payload = dict(raw)
        if is_dataclass(config_cls):
            allowed = {field.name for field in fields(config_cls)}
            canonical_payload = {
                key: value for key, value in raw.items() if str(key) in allowed
            }
        return from_mapping(canonical_payload)
    return config_cls(**dict(raw))


__all__ = [
    "load_canonical_config_from_pretrained",
    "load_pretrained_state_dict",
    "save_canonical_config_bundle",
    "save_pretrained_state_dict",
]
