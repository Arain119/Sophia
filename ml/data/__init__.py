"""Dataset and token-stream primitives used by training/tooling.

The stable public API of this package is the ``token_shards`` subpackage.
Other dataset build and curation logic lives under ``ml.tooling``.
"""

from __future__ import annotations

from . import token_shards


__all__ = ["token_shards"]
