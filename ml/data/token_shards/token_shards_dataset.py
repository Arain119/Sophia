from __future__ import annotations

import os
from collections.abc import Iterator

from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from ml.data.token_shards.shard_manifest import TokenShardManifest
from ml.data.token_shards.shard_manifest import load_manifest

from .token_shards_support import np_dtype
from .token_shards_iterator import TokenStreamIterator


class TokenStreamDataset(IterableDataset):
    """
    Infinite pretrain stream over offline token shards.

    Yields dicts containing:
      - input_ids: [S] or [B,S]
      - labels: same as input_ids
      - labels_is_input_ids: True
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        seq_len: int,
        seed: int = 42,
        batch_size: int = 0,
        shard_preload: int = 0,
        shard_preload_bytes: int = 0,
    ) -> None:
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be >0 (got {seq_len})")
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >=0 (got {batch_size})")
        self.shard_preload = bool(int(shard_preload) == 1)
        self.shard_preload_bytes = max(int(shard_preload_bytes), 0)

        self._manifest = load_manifest(self.manifest_path)
        self._manifest_dir = os.path.dirname(os.path.abspath(self.manifest_path)) or "."
        self._dtype = np_dtype(self._manifest.dtype)
        self._shards = list(self._manifest.shards)
        if not self._shards:
            raise ValueError(
                f"Token shard manifest has no shards: {self.manifest_path}"
            )
        self._block_tokens = (
            int(self.seq_len)
            if int(self.batch_size) <= 0
            else int(self.batch_size) * int(self.seq_len)
        )
        max_shard_tokens = max(int(shard.tokens) for shard in self._shards)
        if int(max_shard_tokens) < int(self._block_tokens):
            raise ValueError(
                "Token shard manifest cannot yield any window: "
                f"manifest={self.manifest_path!r} block_tokens={self._block_tokens} "
                f"max_shard_tokens={max_shard_tokens}"
            )

    @property
    def manifest(self) -> TokenShardManifest:
        return self._manifest

    def __iter__(self) -> Iterator[dict[str, object]]:
        info = get_worker_info()
        worker_id = int(info.id) if info is not None else 0
        num_workers = int(info.num_workers) if info is not None else 1
        return TokenStreamIterator(
            self,
            worker_id=int(worker_id),
            num_workers=max(int(num_workers), 1),
        )


__all__ = ["TokenStreamDataset"]
