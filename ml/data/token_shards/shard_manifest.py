from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass

from .tokenizer_fingerprint import (
    TOKENIZER_BUNDLE_FILES,
    compute_tokenizer_bundle_sha1,
    tokenizer_bundle_fingerprints,
)


@dataclass(frozen=True)
class TokenShard:
    path: str
    tokens: int


@dataclass(frozen=True)
class TokenShardManifest:
    dtype: str
    shards: tuple[TokenShard, ...]
    eos_token_id: int | None = None
    tokenizer_sha1: str = ""

    @property
    def total_tokens(self) -> int:
        return sum(int(s.tokens) for s in self.shards)


def sha1_file(path: str) -> str:
    # SHA1 is used here strictly for fingerprinting (integrity / mismatch diagnostics), not for security.
    # Prefer `usedforsecurity=False` when available (e.g. FIPS builds).
    try:
        h = hashlib.sha1(usedforsecurity=False)
    except TypeError:  # pragma: no cover - older Python/OpenSSL builds
        h = hashlib.sha1()  # nosec B324
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp = path + suffix
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_manifest(path: str) -> TokenShardManifest:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid token shard manifest (expected object): {path}")

    dtype = str(obj.get("dtype") or "").strip()
    if dtype != "int32":
        raise ValueError(
            f"Unsupported dtype={dtype!r} in token shard manifest: {path}. "
            "Sophia only supports int32 token shards."
        )

    shards_obj = obj.get("shards")
    if not isinstance(shards_obj, list) or not shards_obj:
        raise ValueError(
            f"Invalid token shard manifest: `shards` must be a non-empty list: {path}"
        )

    shards: list[TokenShard] = []
    for item in shards_obj:
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid shard entry (expected object) in manifest: {path}"
            )
        spath = str(item.get("path") or "").strip()
        if not spath:
            raise ValueError(
                f"Invalid shard entry (missing `path`) in manifest: {path}"
            )
        tokens = int(item.get("tokens") or 0)
        if tokens <= 0:
            raise ValueError(
                f"Invalid shard entry (tokens<=0) for {spath!r} in manifest: {path}"
            )
        shards.append(TokenShard(path=spath, tokens=tokens))

    eos_token_id = obj.get("eos_token_id")
    eos_token_id = int(eos_token_id) if eos_token_id is not None else None

    tokenizer_sha1 = obj.get("tokenizer_sha1")
    tokenizer_sha1 = (
        str(tokenizer_sha1).strip().lower() if tokenizer_sha1 is not None else ""
    )
    if tokenizer_sha1:
        if len(tokenizer_sha1) != 40 or any(
            c not in "0123456789abcdef" for c in tokenizer_sha1
        ):
            raise ValueError(f"Invalid tokenizer_sha1 in token shard manifest: {path}")

    return TokenShardManifest(
        dtype=dtype,
        shards=tuple(shards),
        eos_token_id=eos_token_id,
        tokenizer_sha1=tokenizer_sha1,
    )


def save_manifest(path: str, manifest: TokenShardManifest) -> None:
    base_dir = os.path.dirname(os.path.abspath(path)) or "."
    shards: list[dict] = []
    for s in manifest.shards:
        rel = os.path.relpath(os.path.abspath(os.path.join(base_dir, s.path)), base_dir)
        shards.append({"path": rel.replace("\\", "/"), "tokens": int(s.tokens)})

    payload = {
        "dtype": str(manifest.dtype),
        "eos_token_id": manifest.eos_token_id,
        "tokenizer_sha1": manifest.tokenizer_sha1,
        "total_tokens": int(manifest.total_tokens),
        "shards": shards,
    }
    write_json_atomic(path, payload)


def validate_tokenizer_fingerprint(
    *, tokenizer_dir: str, expected_sha1: str, allow_empty: bool = False
) -> None:
    expected = str(expected_sha1 or "").strip().lower()
    if not expected:
        if bool(allow_empty):
            return
        raise ValueError(
            "Missing tokenizer_sha1 in manifest.\n"
            "Rebuild shards with: python -m ml.training.pretrain.shard_builder."
        )
    if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid tokenizer sha1 (expected 40 hex chars)")

    tok_dir = os.path.abspath(str(tokenizer_dir))
    computed_bundle = compute_tokenizer_bundle_sha1(tok_dir).lower()
    if computed_bundle == expected:
        return

    components: list[str] = []
    for fp in tokenizer_bundle_fingerprints(tok_dir):
        components.append(f"{fp.label} sha1={sha1_file(fp.path).lower()}")

    raise ValueError(
        "Tokenizer sha1 mismatch.\n"
        f"expected={expected_sha1}\n"
        f"computed_bundle_sha1={computed_bundle}\n"
        + "\n".join(components)
        + "\nExpected tokenizer bundle files:\n"
        + "\n".join(f"- {x}" for x in TOKENIZER_BUNDLE_FILES)
    )
