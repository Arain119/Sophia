"""Stable token-shard public API.

These helpers are shared across shard builders, dataset materialization,
training resume logic, and tooling audits.
"""

from ml.data.token_shards.shard_manifest import (
    TokenShard,
    TokenShardManifest,
    load_manifest,
    save_manifest,
    sha1_file,
    validate_tokenizer_fingerprint,
    write_json_atomic,
)
from ml.data.token_shards.token_shards_dataset import TokenStreamDataset
from ml.data.token_shards.tokenizer_fingerprint import (
    CHAT_TEMPLATE_JINJA,
    TOKENIZER_BUNDLE_FILES,
    TOKENIZER_CONFIG_JSON,
    TOKENIZER_JSON,
    TokenizerFingerprint,
    compute_tokenizer_bundle_sha1,
    copy_tokenizer_bundle,
    dataset_root_from_manifest_path,
    dataset_tokenizer_dir_for_manifest,
    preferred_fingerprint,
    resolve_matching_tokenizer_dir,
    tokenizer_bundle_fingerprints,
    tokenizer_bundle_matches_sha1,
)

__all__ = [
    "CHAT_TEMPLATE_JINJA",
    "TOKENIZER_BUNDLE_FILES",
    "TOKENIZER_CONFIG_JSON",
    "TOKENIZER_JSON",
    "TokenShard",
    "TokenShardManifest",
    "TokenStreamDataset",
    "TokenizerFingerprint",
    "compute_tokenizer_bundle_sha1",
    "copy_tokenizer_bundle",
    "dataset_root_from_manifest_path",
    "dataset_tokenizer_dir_for_manifest",
    "load_manifest",
    "preferred_fingerprint",
    "resolve_matching_tokenizer_dir",
    "save_manifest",
    "sha1_file",
    "tokenizer_bundle_fingerprints",
    "tokenizer_bundle_matches_sha1",
    "validate_tokenizer_fingerprint",
    "write_json_atomic",
]
