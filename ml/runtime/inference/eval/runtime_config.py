from __future__ import annotations

from dataclasses import dataclass, replace

from ml.errors import SophiaUsageError
from ml.runtime.inference.eval.common import require_cuda, setup_seed, setup_torch_backends
from ml.core.common.mapping import object_mapping


@dataclass(frozen=True)
class EvalRuntimeConfig:
    save_dir: str = "out"
    export_dir: str = ""
    seed: int = 2026
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.8
    do_sample: bool = True
    top_k: int = 0
    max_cache_len: int = 0
    history_turns: int = 0
    device: str = "cuda:0"
    mode: str = "ask"


def eval_runtime_config_from_args(args: object) -> EvalRuntimeConfig:
    payload = object_mapping(args)
    return EvalRuntimeConfig(
        save_dir=str(payload.get("save_dir", "out") or "out"),
        export_dir=str(payload.get("export_dir", "") or ""),
        seed=int(payload.get("seed", 2026)),
        max_new_tokens=int(payload.get("max_new_tokens", 256)),
        temperature=float(payload.get("temperature", 0.6)),
        top_p=float(payload.get("top_p", 0.8)),
        do_sample=bool(int(payload.get("do_sample", 1))),
        top_k=int(payload.get("top_k", 0)),
        max_cache_len=int(payload.get("max_cache_len", 0)),
        history_turns=int(payload.get("history_turns", 0)),
        device=str(payload.get("device", "cuda:0") or "cuda:0"),
        mode=str(payload.get("mode", "ask") or "ask"),
    )


def resolve_eval_runtime_config(config: EvalRuntimeConfig) -> EvalRuntimeConfig:
    requested_device = str(config.device or "cuda:0").strip().lower()
    resolved_device = "cpu" if requested_device == "cpu" else require_cuda(str(config.device))
    setup_torch_backends()

    if int(config.history_turns) < 0:
        raise SophiaUsageError("[ERR] --history_turns must be >= 0")
    if int(config.max_new_tokens) < 0:
        raise SophiaUsageError("[ERR] --max_new_tokens must be >= 0")
    if bool(config.do_sample) and float(config.temperature) <= 0.0:
        raise SophiaUsageError("[ERR] --temperature must be > 0 when --do_sample=1")
    if not bool(config.do_sample) and float(config.temperature) < 0.0:
        raise SophiaUsageError("[ERR] --temperature must be >= 0 when --do_sample=0")
    if not (0.0 < float(config.top_p) <= 1.0):
        raise SophiaUsageError("[ERR] --top_p must be in (0, 1].")
    if int(config.top_k) < 0:
        raise SophiaUsageError("[ERR] --top_k must be >= 0")
    if int(config.max_cache_len) < 0:
        raise SophiaUsageError("[ERR] --max_cache_len must be >= 0")

    resolved = replace(config, device=str(resolved_device))
    if int(resolved.history_turns) % 2 != 0:
        rounded_history_turns = max(int(resolved.history_turns) - 1, 0)
        print(
            "[WARN] history_turns must be even; rounding down: "
            f"{resolved.history_turns} -> {rounded_history_turns}"
        )
        resolved = replace(resolved, history_turns=int(rounded_history_turns))

    if int(resolved.seed) >= 0:
        setup_seed(int(resolved.seed))
    return resolved


__all__ = [
    "EvalRuntimeConfig",
    "eval_runtime_config_from_args",
    "resolve_eval_runtime_config",
]
