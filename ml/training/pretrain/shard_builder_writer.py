from __future__ import annotations

import os
import uuid
from collections.abc import Sequence

import numpy as np

from ml.data.token_shards.shard_manifest import TokenShard


class ShardWriter:
    def __init__(self, out_dir: str, *, shard_size_tokens: int, dtype: np.dtype):
        self.out_dir = str(out_dir)
        self.shard_size_tokens = max(int(shard_size_tokens), 1024)
        self.dtype = np.dtype(dtype)

        self._fp = None
        self._tmp_path = None
        self._final_path = None
        self._shard_idx = 0
        self._tokens_in_current = 0
        self.shards: list[TokenShard] = []

    def _open_new(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        final_path = os.path.join(self.out_dir, f"shard_{self._shard_idx:05d}.bin")
        suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
        tmp_path = final_path + suffix
        self._fp = open(tmp_path, "wb")
        self._tmp_path = tmp_path
        self._final_path = final_path
        self._tokens_in_current = 0

    def _close_current(self) -> None:
        if self._fp is None:
            return
        self._fp.flush()
        self._fp.close()
        self._fp = None

        if self._tmp_path and self._final_path:
            os.replace(self._tmp_path, self._final_path)
            self.shards.append(
                TokenShard(
                    path=os.path.basename(self._final_path),
                    tokens=int(self._tokens_in_current),
                )
            )
        self._tmp_path = None
        self._final_path = None
        self._tokens_in_current = 0
        self._shard_idx += 1

    def write(self, tokens: np.ndarray) -> None:
        if tokens is None:
            return
        if not isinstance(tokens, np.ndarray):
            tokens = np.asarray(tokens, dtype=self.dtype)
        if tokens.size <= 0:
            return
        if tokens.dtype != self.dtype:
            tokens = tokens.astype(self.dtype, copy=False)

        idx = 0
        n = int(tokens.size)
        while idx < n:
            if self._fp is None:
                self._open_new()
            if self._fp is None:
                raise RuntimeError(
                    "Shard writer is not initialized (failed to open output shard file)."
                )
            space = int(self.shard_size_tokens - self._tokens_in_current)
            take = min(space, n - idx)
            if take <= 0:
                self._close_current()
                continue
            chunk = tokens[idx : idx + take]
            self._fp.write(chunk.tobytes(order="C"))
            idx += take
            self._tokens_in_current += take
            if self._tokens_in_current >= self.shard_size_tokens:
                self._close_current()

    def close(self) -> None:
        if self._fp is not None and self._tokens_in_current > 0:
            self._close_current()
        else:
            # Ensure temp file does not linger on zero-length shards.
            if self._fp is not None:
                try:
                    self._fp.close()
                finally:
                    self._fp = None
            if self._tmp_path and os.path.exists(self._tmp_path):
                os.remove(self._tmp_path)

    def flush_shard(self) -> None:
        """Finalize the current partial shard so the next write opens a fresh one."""
        if self._fp is not None and self._tokens_in_current > 0:
            self._close_current()

    def seed_resume(self, *, shard_idx: int, shards: Sequence[TokenShard]) -> None:
        """Restore writer state when continuing a previously interrupted build."""
        if self._fp is not None:
            raise RuntimeError("cannot seed_resume a writer with an open shard")
        self._shard_idx = int(shard_idx)
        self.shards = list(shards)


__all__ = ["ShardWriter"]
