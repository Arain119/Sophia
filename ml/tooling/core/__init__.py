"""
Stable shared utilities for Sophia tooling.

Keep this layer small and dependency-light so production maintenance commands can
reuse it without importing large script modules.
"""

from ml.tooling.core.json_io import load_json_dict, write_json_atomic
from ml.data.jsonl_stream import (
    JsonlFieldStats,
    JsonlReadStats,
    iter_jsonl_field,
    iter_jsonl_objects,
)
from ml.tooling.core.parquet_inventory import (
    ParquetRepresentationFamily,
    choose_canonical_parquet_paths,
    is_reshard_name,
    list_glob_prefer_plain,
    list_tree_parquets_prefer_plain,
    parquet_family_base_name,
    representation_families_for_paths,
)
from ml.tooling.core.path_safety import (
    remove_tree_checked,
    require_descendant,
    require_safe_path,
    require_separate_trees,
    swap_directory_trees,
)
from ml.data.pretrain_filters import (
    has_repeated_sentences,
    is_poison_repetitive_text,
    normalize_for_poison_scan,
    normalize_text,
)
from ml.tooling.core.pretrain_mix_presets import (
    PRESETS,
    BucketMode,
    PretrainMixPreset,
    SourcePreset,
    get_preset,
)
from ml.tooling.core.pretrain_taxonomy_user import (
    LABELS,
    infer_user_taxonomy_label,
)
from ml.tooling.core.text_hygiene import (
    has_emoji,
    has_human_bot_tags,
    has_placeholders,
    mentions_ai,
    strip_think_blocks,
)

__all__ = [
    "JsonlFieldStats",
    "JsonlReadStats",
    "LABELS",
    "PRESETS",
    "ParquetRepresentationFamily",
    "PretrainMixPreset",
    "SourcePreset",
    "BucketMode",
    "choose_canonical_parquet_paths",
    "get_preset",
    "has_repeated_sentences",
    "has_emoji",
    "has_human_bot_tags",
    "has_placeholders",
    "infer_user_taxonomy_label",
    "is_reshard_name",
    "is_poison_repetitive_text",
    "iter_jsonl_field",
    "iter_jsonl_objects",
    "list_glob_prefer_plain",
    "list_tree_parquets_prefer_plain",
    "load_json_dict",
    "mentions_ai",
    "normalize_text",
    "normalize_for_poison_scan",
    "parquet_family_base_name",
    "remove_tree_checked",
    "representation_families_for_paths",
    "require_descendant",
    "require_safe_path",
    "require_separate_trees",
    "strip_think_blocks",
    "swap_directory_trees",
    "write_json_atomic",
]
