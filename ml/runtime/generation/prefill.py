from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol
import uuid

import torch

from ml.runtime.generation.prefix_cache import (
    PrefixCache,
    PrefixCacheSnapshotError,
    PrefixSnapshot,
    model_prefix_cache_namespace as model_cache_namespace,
)
from ml.runtime.model.state import RuntimeCacheSnapshot


class PrefillRuntime(Protocol):
    def reset_runtime_cache(self) -> None: ...

    def cache_dump_prefix(
        self,
        device: str = "cpu",
        *,
        prefix_len: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot: ...

    def runtime_max_seq_len(self) -> int: ...

    def runtime_prefix_replay_len(self) -> int: ...

    def replay_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def cache_load(self, snapshot: RuntimeCacheSnapshot) -> None: ...

    def recompute_state_cache(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


def prefix_cache_key(tokens: list[int], *, namespace: str) -> str:
    payload = f"{namespace}\n" + ",".join(str(int(token)) for token in tokens)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def prefix_cache_path(
    cache_dir: str | Path,
    tokens: list[int],
    *,
    namespace: str,
) -> Path:
    return Path(cache_dir) / f"{prefix_cache_key(tokens, namespace=namespace)}.pt"


def validate_snapshot_payload(payload: object, *, path: Path) -> PrefixSnapshot:
    if not isinstance(payload, dict):
        raise PrefixCacheSnapshotError(
            f"prefix cache snapshot at {path} must be a dict"
        )
    required = {"prefix_len", "replay_len", "cache"}
    missing = required.difference(payload)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise PrefixCacheSnapshotError(
            f"prefix cache snapshot at {path} is missing required keys: {missing_str}"
        )
    cache = payload["cache"]
    if not isinstance(cache, dict):
        raise PrefixCacheSnapshotError(
            f"prefix cache snapshot at {path} has non-dict cache payload"
        )
    prefix_len = int(payload["prefix_len"])
    replay_len = int(payload["replay_len"])
    if prefix_len < 0:
        raise PrefixCacheSnapshotError(
            f"prefix cache snapshot at {path} has negative prefix_len: {prefix_len}"
        )
    if replay_len < 0:
        raise PrefixCacheSnapshotError(
            f"prefix cache snapshot at {path} has negative replay_len: {replay_len}"
        )
    try:
        cache_snapshot = RuntimeCacheSnapshot.from_payload(cache)
    except Exception as exc:  # noqa: BLE001
        raise PrefixCacheSnapshotError(
            f"prefix cache snapshot at {path} has invalid runtime cache payload"
        ) from exc
    return PrefixSnapshot(
        prefix_len=prefix_len,
        cache=cache_snapshot,
        replay_len=replay_len,
    )


def load_snapshot_from_disk(
    cache_dir: str | Path,
    tokens: list[int],
    *,
    min_tokens: int,
    namespace: str,
) -> tuple[int, PrefixSnapshot | None]:
    cache_root = Path(cache_dir)
    if not cache_root.exists():
        return 0, None
    if len(tokens) < int(min_tokens):
        return 0, None

    for prefix_len in range(len(tokens), int(min_tokens) - 1, -1):
        path = prefix_cache_path(
            cache_root,
            tokens[:prefix_len],
            namespace=namespace,
        )
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise PrefixCacheSnapshotError(
                f"failed to load prefix cache snapshot at {path}"
            ) from exc
        snapshot = validate_snapshot_payload(payload, path=path)
        return prefix_len, snapshot
    return 0, None


def save_snapshot_to_disk(
    cache_dir: str | Path,
    tokens: list[int],
    snapshot: PrefixSnapshot,
    *,
    namespace: str,
) -> None:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = prefix_cache_path(cache_root, tokens, namespace=namespace)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    torch.save(
        {
            "prefix_len": int(snapshot.prefix_len),
            "replay_len": int(snapshot.replay_len),
            "cache": snapshot.cache.to_payload(),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def build_prefix_snapshot(model: PrefillRuntime, prefix_len: int) -> PrefixSnapshot:
    cache_snapshot = model.cache_dump_prefix(
        device="cpu",
        prefix_len=int(prefix_len),
        batch_size=1,
    )
    return PrefixSnapshot(
        prefix_len=int(prefix_len),
        cache=cache_snapshot,
        replay_len=int(model.runtime_prefix_replay_len()),
    )


def store_prefix_snapshot(
    model: PrefillRuntime,
    *,
    tokens: list[int],
    prefix_cache: PrefixCache | None,
    prefix_cache_dir: str | Path | None,
) -> None:
    snapshot = build_prefix_snapshot(model, len(tokens))
    namespace = model_cache_namespace(model)
    if prefix_cache is not None:
        prefix_cache.put(tokens, snapshot, namespace=namespace)
    if prefix_cache_dir is not None:
        save_snapshot_to_disk(
            prefix_cache_dir,
            tokens,
            snapshot,
            namespace=namespace,
        )


def resolve_prefix_snapshot(
    *,
    model: PrefillRuntime,
    tokens: list[int],
    prefix_cache: PrefixCache | None,
    prefix_cache_dir: str | Path | None,
    prefix_cache_min_tokens: int,
) -> tuple[int, PrefixSnapshot | None]:
    namespace = model_cache_namespace(model)
    match_len, snapshot = (
        prefix_cache.get_longest_prefix(
            tokens,
            namespace=namespace,
            min_tokens=int(prefix_cache_min_tokens),
        )
        if prefix_cache is not None
        else (0, None)
    )
    if snapshot is not None or prefix_cache_dir is None:
        return int(match_len), snapshot

    disk_match_len, disk_snapshot = load_snapshot_from_disk(
        prefix_cache_dir,
        tokens,
        min_tokens=int(prefix_cache_min_tokens),
        namespace=namespace,
    )
    if disk_snapshot is not None and prefix_cache is not None:
        prefix_cache.put(
            tokens[: int(disk_match_len)],
            disk_snapshot,
            namespace=namespace,
        )
    return int(disk_match_len), disk_snapshot


def maybe_reset_model_cache(model: PrefillRuntime) -> None:
    model.reset_runtime_cache()


def resolve_prefill_chunk_size(
    model: PrefillRuntime,
    prompt_len: int,
    prefill_chunk_size: int | None,
) -> int:
    if prefill_chunk_size is not None:
        return int(prefill_chunk_size)
    max_seq_len = int(model.runtime_max_seq_len())
    return max(1, min(max_seq_len, int(prompt_len)))


def replay_prompt_range(
    model: PrefillRuntime,
    *,
    input_ids: torch.Tensor,
    start_pos: int,
    end_pos: int,
    prefill_chunk_size: int | None,
) -> torch.Tensor:
    if int(end_pos) <= int(start_pos):
        raise ValueError(
            f"end_pos must be > start_pos, got start_pos={start_pos}, end_pos={end_pos}"
        )
    if prefill_chunk_size is None:
        logits, _ = model.replay_with_cache(
            input_ids[:, int(start_pos) : int(end_pos)],
            start_pos=int(start_pos),
            return_all_logits=False,
        )
        return logits

    chunk_size = resolve_prefill_chunk_size(
        model,
        int(end_pos) - int(start_pos),
        prefill_chunk_size,
    )
    logits: torch.Tensor | None = None
    for chunk_start in range(int(start_pos), int(end_pos), int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), int(end_pos))
        logits, _ = model.replay_with_cache(
            input_ids[:, chunk_start:chunk_end],
            start_pos=int(chunk_start),
            return_all_logits=False,
        )
    if logits is None:
        raise RuntimeError("failed to replay prompt range")
    return logits


def replay_prefill_snapshot(
    model: PrefillRuntime,
    *,
    snapshot: PrefixSnapshot,
    match_len: int,
    prompt_len: int,
    tokens: list[int],
    input_ids: torch.Tensor,
    model_device: torch.device,
    prefill_chunk_size: int | None,
) -> torch.Tensor:
    if int(snapshot.prefix_len) != int(match_len):
        raise ValueError(
            "prefix cache snapshot length mismatch: "
            f"expected {match_len}, got {snapshot.prefix_len}"
        )
    model.cache_load(snapshot.cache)
    replay_len = min(int(snapshot.replay_len), int(match_len))
    replay_start = int(match_len) - replay_len
    replay_tokens = torch.tensor(
        [tokens[replay_start:match_len]],
        dtype=torch.long,
        device=model_device,
    )
    prefill_logits, _ = model.recompute_state_cache(
        replay_tokens,
        start_pos=int(replay_start),
    )
    if int(match_len) >= int(prompt_len):
        return prefill_logits
    return replay_prompt_range(
        model,
        input_ids=input_ids,
        start_pos=int(match_len),
        end_pos=int(prompt_len),
        prefill_chunk_size=prefill_chunk_size,
    )


def prefill_prompt(
    model: PrefillRuntime,
    *,
    input_ids: torch.Tensor,
    input_ids_list: list[list[int]] | None,
    model_device: torch.device,
    prefix_cache: PrefixCache | None,
    prefix_cache_min_tokens: int,
    prefix_cache_dir: str | Path | None,
    prefill_chunk_size: int | None,
) -> torch.Tensor:
    prompt_len = int(input_ids.size(1))
    prefix_cache_enabled = prefix_cache is not None or prefix_cache_dir is not None
    prefill_logits: torch.Tensor | None = None
    tokens: list[int] | None = None

    if prefix_cache_enabled:
        tokens = (
            input_ids_list[0]
            if input_ids_list is not None
            else input_ids[0].detach().cpu().tolist()
        )

    if prefix_cache_enabled:
        assert tokens is not None
        match_len, snapshot = resolve_prefix_snapshot(
            model=model,
            tokens=tokens,
            prefix_cache=prefix_cache,
            prefix_cache_dir=prefix_cache_dir,
            prefix_cache_min_tokens=int(prefix_cache_min_tokens),
        )
        if snapshot is not None and int(match_len) >= int(prefix_cache_min_tokens):
            try:
                prefill_logits = replay_prefill_snapshot(
                    model,
                    snapshot=snapshot,
                    match_len=int(match_len),
                    prompt_len=int(prompt_len),
                    tokens=tokens,
                    input_ids=input_ids,
                    model_device=model_device,
                    prefill_chunk_size=prefill_chunk_size,
                )
            except Exception as exc:
                maybe_reset_model_cache(model)
                raise RuntimeError(
                    "prefix cache snapshot replay failed; the cached state is incompatible "
                    "with the current model/runtime contract"
                ) from exc
            store_prefix_snapshot(
                model,
                tokens=tokens,
                prefix_cache=prefix_cache,
                prefix_cache_dir=prefix_cache_dir,
            )

    if prefill_logits is not None:
        return prefill_logits

    prefill_logits = replay_prompt_range(
        model,
        input_ids=input_ids,
        start_pos=0,
        end_pos=int(prompt_len),
        prefill_chunk_size=prefill_chunk_size,
    )

    if (
        prefix_cache_enabled
        and tokens is not None
        and len(tokens) >= int(prefix_cache_min_tokens)
    ):
        store_prefix_snapshot(
            model,
            tokens=tokens,
            prefix_cache=prefix_cache,
            prefix_cache_dir=prefix_cache_dir,
        )

    if prefill_logits is None:
        raise RuntimeError("failed to compute prompt logits before generation")
    return prefill_logits


__all__ = [
    "PrefillRuntime",
    "build_prefix_snapshot",
    "load_snapshot_from_disk",
    "maybe_reset_model_cache",
    "model_cache_namespace",
    "prefill_prompt",
    "prefix_cache_key",
    "prefix_cache_path",
    "replay_prompt_range",
    "resolve_prefill_chunk_size",
    "save_snapshot_to_disk",
    "store_prefix_snapshot",
    "validate_snapshot_payload",
]
