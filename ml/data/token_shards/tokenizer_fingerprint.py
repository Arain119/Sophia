from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


TOKENIZER_JSON = "tokenizer.json"
TOKENIZER_CONFIG_JSON = "tokenizer_config.json"
CHAT_TEMPLATE_JINJA = "chat_template.jinja"
TOKENIZER_BUNDLE_FILES = (TOKENIZER_JSON, TOKENIZER_CONFIG_JSON, CHAT_TEMPLATE_JINJA)
_PRETRAIN_SPLIT_NAMES = frozenset({"train", "val", "test"})


@dataclass(frozen=True)
class TokenizerFingerprint:
    """
    A tokenizer fingerprint file used for manifest verification.

    Sophia is pinned to a tokenizer bundle:
    - tokenizer.json (fast tokenizer)
    - tokenizer_config.json (tokenizer behavior settings)
    - chat_template.jinja (optional chat formatting template)
    """

    label: str
    path: str


def _sha1_new():
    # SHA1 is used here strictly for fingerprinting (integrity / mismatch diagnostics), not for security.
    # Prefer `usedforsecurity=False` when available (e.g. FIPS builds).
    try:
        return hashlib.sha1(usedforsecurity=False)
    except TypeError:  # pragma: no cover - older Python/OpenSSL builds
        return hashlib.sha1()  # nosec B324


def tokenizer_bundle_fingerprints(tokenizer_dir: str) -> list[TokenizerFingerprint]:
    tok_dir = os.path.abspath(str(tokenizer_dir))
    out: list[TokenizerFingerprint] = []
    missing: list[str] = []
    for name in TOKENIZER_BUNDLE_FILES:
        path = os.path.join(tok_dir, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        out.append(TokenizerFingerprint(label=name, path=path))
    if missing:
        raise ValueError(
            f"Tokenizer directory missing required files: {tok_dir}\n"
            + "\n".join(f"- {m}" for m in missing)
        )
    return out


def compute_tokenizer_bundle_sha1(tokenizer_dir: str) -> str:
    fps = tokenizer_bundle_fingerprints(tokenizer_dir)
    h = _sha1_new()
    for fp in fps:
        h.update(fp.label.encode("utf-8"))
        h.update(b"\0")
        with open(fp.path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def preferred_fingerprint(tokenizer_dir: str) -> TokenizerFingerprint:
    for fp in tokenizer_bundle_fingerprints(tokenizer_dir):
        if fp.label == TOKENIZER_JSON:
            return fp
    tok_dir = os.path.abspath(str(tokenizer_dir))
    raise RuntimeError(f"Missing {TOKENIZER_JSON} under: {tok_dir}")


def dataset_root_from_manifest_path(manifest_path: str) -> str:
    manifest = Path(manifest_path).expanduser().resolve()
    split_dir = manifest.parent
    if split_dir.name.lower() in _PRETRAIN_SPLIT_NAMES:
        return str(split_dir.parent)
    return str(split_dir)


def _dataset_tokenizer_candidates(manifest_path: str) -> list[str]:
    dataset_root = Path(dataset_root_from_manifest_path(manifest_path))
    candidates: list[Path] = []
    dataset_meta = dataset_root / "dataset.json"
    if dataset_meta.exists():
        try:
            obj = json.loads(dataset_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            obj = None
        if isinstance(obj, dict):
            tokenizer_meta = obj.get("tokenizer")
            if isinstance(tokenizer_meta, dict):
                rel = str(tokenizer_meta.get("dir") or "").strip()
                if rel:
                    candidates.append(dataset_root / rel)
            elif isinstance(tokenizer_meta, str):
                rel = str(tokenizer_meta).strip()
                if rel:
                    candidates.append(dataset_root / rel)
    candidates.append(dataset_root / "tokenizer")

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = os.path.abspath(str(candidate))
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def dataset_tokenizer_dir_for_manifest(manifest_path: str) -> str | None:
    for candidate in _dataset_tokenizer_candidates(manifest_path):
        try:
            tokenizer_bundle_fingerprints(candidate)
        except ValueError:
            continue
        return os.path.abspath(str(candidate))
    return None


def tokenizer_bundle_matches_sha1(*, tokenizer_dir: str, expected_sha1: str) -> bool:
    expected = str(expected_sha1 or "").strip().lower()
    if not expected:
        return False
    try:
        computed = compute_tokenizer_bundle_sha1(tokenizer_dir).lower()
    except (OSError, ValueError):
        # tokenizer bundle is missing required files
        return False
    return computed == expected


def default_modeling_tokenizer_dir() -> str:
    """Absolute path to the in-package tokenizer bundle (`modeling/text`).

    This is the canonical default tokenizer directory for shard audit / alignment
    / decode tooling. Resolved relative to this package (not the repo root) so it
    works for both editable checkouts and installed wheels.
    """
    # tokenizer_fingerprint.py lives at ml/data/token_shards/; the bundle
    # lives at ml/modeling/text/.
    return str(Path(__file__).resolve().parents[2] / "modeling" / "text")


def resolve_matching_tokenizer_dir(
    *,
    manifest_path: str,
    expected_sha1: str,
    explicit_tokenizer_dir: str = "",
    default_tokenizer_dir: str = "",
) -> str:
    explicit = str(explicit_tokenizer_dir or "").strip()
    default_dir = str(default_tokenizer_dir or "").strip()
    dataset_tokenizer = dataset_tokenizer_dir_for_manifest(manifest_path) if not explicit else None

    candidates: list[str] = []
    if explicit:
        candidates.append(os.path.abspath(explicit))
    else:
        if dataset_tokenizer:
            candidates.append(os.path.abspath(dataset_tokenizer))
        if default_dir:
            candidates.append(os.path.abspath(default_dir))

    for candidate in candidates:
        if tokenizer_bundle_matches_sha1(
            tokenizer_dir=candidate,
            expected_sha1=expected_sha1,
        ):
            return os.path.abspath(candidate)

    if explicit:
        return os.path.abspath(explicit)
    if dataset_tokenizer:
        return os.path.abspath(dataset_tokenizer)
    if default_dir:
        return os.path.abspath(default_dir)
    raise ValueError(
        "Unable to resolve a tokenizer bundle for the dataset manifest.\n"
        f"manifest_path={os.path.abspath(str(manifest_path))}\n"
        f"expected_tokenizer_sha1={str(expected_sha1 or '').strip().lower()}\n"
        "Provide --tokenizer_path explicitly, or place a matching tokenizer bundle under the dataset root."
    )


def copy_tokenizer_bundle(
    *,
    tokenizer_dir: str,
    out_dir: str,
    expected_sha1: str = "",
) -> str:
    src_dir = os.path.abspath(str(tokenizer_dir))
    dst_dir = os.path.abspath(str(out_dir))
    if expected_sha1 and not tokenizer_bundle_matches_sha1(
        tokenizer_dir=src_dir,
        expected_sha1=expected_sha1,
    ):
        raise ValueError(
            "Tokenizer sha1 mismatch for dataset-local bundle copy.\n"
            f"expected={expected_sha1}\n"
            f"source_dir={src_dir}\n"
            f"computed={compute_tokenizer_bundle_sha1(src_dir).lower()}"
        )

    fps = tokenizer_bundle_fingerprints(src_dir)
    os.makedirs(dst_dir, exist_ok=True)
    for fp in fps:
        shutil.copy2(fp.path, os.path.join(dst_dir, fp.label))
    return dst_dir
