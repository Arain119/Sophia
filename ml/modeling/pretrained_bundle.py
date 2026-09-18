from __future__ import annotations

from dataclasses import is_dataclass
from dataclasses import asdict
import json
import os

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file


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
    export_state: dict[str, torch.Tensor] = {}
    seen_cpu_storages: set[tuple[int, int]] = set()
    for key, value in dict(state_dict).items():
        tensor = value.detach().cpu()
        storage = tensor.untyped_storage()
        storage_key = (int(storage.data_ptr()), int(storage.nbytes()))
        if storage_key in seen_cpu_storages:
            tensor = tensor.clone(memory_format=torch.preserve_format)
            storage = tensor.untyped_storage()
            storage_key = (int(storage.data_ptr()), int(storage.nbytes()))
        seen_cpu_storages.add(storage_key)
        export_state[str(key)] = tensor
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
) -> dict[str, torch.Tensor]:
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
    raise RuntimeError(
        "exported model directory has no supported model weights: "
        f"{resolved_export_dir}"
    )


def load_canonical_config_from_pretrained[ConfigT](
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
    from_pretrained_mapping = getattr(config_cls, "from_pretrained_mapping", None)
    if callable(from_pretrained_mapping):
        return from_pretrained_mapping(dict(raw))
    from_mapping = getattr(config_cls, "from_mapping", None)
    if callable(from_mapping):
        return from_mapping(dict(raw))
    return config_cls(**dict(raw))


__all__ = [
    "load_canonical_config_from_pretrained",
    "load_pretrained_state_dict",
    "save_canonical_config_bundle",
    "save_pretrained_state_dict",
]
