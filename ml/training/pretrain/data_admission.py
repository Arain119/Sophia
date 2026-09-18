from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from ml.data.token_shards.shard_manifest import load_manifest, sha1_file
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.errors import SophiaUsageError


POLICY_SCHEMA = "sophia_pretrain_data_admission_policy_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ADMISSION_POLICY = (
    REPO_ROOT / "configs/data/pretrain_data_admission_policy.json"
)
DATASET_SCHEMA = "sophia_pretrain_token_dataset_v1"
DATASET_MARKER = ".sophia_pretrain_token_dataset.json"


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SophiaUsageError(
            f"[ERR] unable to load pretrain data admission policy: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or str(payload.get("schema")) != POLICY_SCHEMA:
        raise SophiaUsageError(
            f"[ERR] unsupported pretrain data admission policy: {path}"
        )
    if bool(payload.get("allow_legacy_assets_as_training_input")):
        raise SophiaUsageError(
            "[ERR] pretrain data admission policy must reject legacy training inputs."
        )
    return payload


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_root(data_path: str) -> Path:
    path = Path(str(data_path)).expanduser().resolve()
    if path.name == "manifest.json":
        path = path.parent
    if path.name.lower() in {"train", "val", "test"}:
        path = path.parent
    return path


def _resolve_relative_file(*, root: Path, value: object, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise SophiaUsageError(f"[ERR] production dataset is missing {label}.")
    path = (root / raw).resolve()
    if root != path and root not in path.parents:
        raise SophiaUsageError(f"[ERR] production dataset {label} escapes its root: {path}")
    if not path.is_file():
        raise SophiaUsageError(f"[ERR] production dataset {label} is missing: {path}")
    return path


def _normalized_quality_proxy_exemptions(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise SophiaUsageError(
            "[ERR] Chinese quality proxy curated exemptions must be a list."
        )
    normalized: list[dict[str, str]] = []
    for row in value:
        if not isinstance(row, dict):
            raise SophiaUsageError(
                "[ERR] Chinese quality proxy curated exemptions must be objects."
            )
        normalized.append(
            {
                "repo_id": str(row.get("repo_id") or "").casefold(),
                "revision": str(row.get("revision") or "").lower(),
                "reason": str(row.get("reason") or ""),
                "evidence_report_sha256": str(
                    row.get("evidence_report_sha256") or ""
                ).lower(),
            }
        )
    return normalized


def _validate_chinese_quality_proxy_lineage(
    *,
    root: Path,
    policy: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    config = policy.get("chinese_quality_proxy")
    if not isinstance(config, dict):
        return
    if not bool(config.get("required")):
        raise SophiaUsageError(
            "[ERR] configured Chinese quality proxy must be required."
        )
    quality_item = artifacts.get("chinese_quality_proxy_report")
    cleaning_item = artifacts.get("cleaning_report")
    if not isinstance(quality_item, dict) or not isinstance(cleaning_item, dict):
        raise SophiaUsageError(
            "[ERR] production dataset is missing Chinese quality proxy lineage."
        )
    quality_path = _resolve_relative_file(
        root=root,
        value=quality_item.get("path"),
        label="Chinese quality proxy report",
    )
    cleaning_path = _resolve_relative_file(
        root=root,
        value=cleaning_item.get("path"),
        label="cleaning report",
    )
    quality = _load_policy_like_json(quality_path, label="Chinese quality proxy report")
    cleaning = _load_policy_like_json(cleaning_path, label="cleaning report")
    if (
        str(quality.get("schema")) != "sophia_chinese_quality_proxy_report_v2"
        or str(quality.get("status")) != "pass"
    ):
        raise SophiaUsageError(
            "[ERR] production Chinese quality proxy report did not pass."
        )
    gate_results = quality.get("gate_results")
    if not isinstance(gate_results, dict) or not gate_results or not all(
        value is True for value in gate_results.values()
    ):
        raise SophiaUsageError(
            "[ERR] production Chinese quality proxy gates are incomplete."
        )
    model = quality.get("model")
    source = quality.get("source")
    if not isinstance(model, dict) or not isinstance(source, dict):
        raise SophiaUsageError(
            "[ERR] production Chinese quality proxy lineage is incomplete."
        )
    validated_threshold = float(model.get("threshold", -1.0))
    policy_threshold = float(config.get("minimum_score", -1.0))
    cleaning_threshold = float(
        cleaning.get("chinese_quality_proxy_minimum_score", -1.0)
    )
    policy_cjk_ratio = float(config.get("minimum_cjk_ratio", -1.0))
    cleaning_cjk_ratio = float(
        cleaning.get("chinese_quality_proxy_minimum_cjk_ratio", -1.0)
    )
    if (
        not validated_threshold <= policy_threshold <= 1.0
        or cleaning_threshold != policy_threshold
        or cleaning_cjk_ratio != policy_cjk_ratio
    ):
        raise SophiaUsageError(
            "[ERR] production Chinese quality proxy thresholds do not match policy."
        )
    if (
        str(cleaning.get("chinese_quality_proxy_report_sha256") or "")
        != str(quality_item.get("sha256") or "")
        or str(cleaning.get("chinese_quality_proxy_model_sha256") or "")
        != str(model.get("sha256") or "")
        or str(cleaning.get("chinese_quality_proxy_labels_sha256") or "")
        != str(source.get("labels_sha256") or "")
    ):
        raise SophiaUsageError(
            "[ERR] production Chinese quality proxy fingerprints do not match cleaning."
        )
    policy_exemptions = _normalized_quality_proxy_exemptions(
        config.get("curated_source_exemptions", [])
    )
    cleaning_exemptions = _normalized_quality_proxy_exemptions(
        cleaning.get("chinese_quality_proxy_curated_source_exemptions", [])
    )
    if cleaning_exemptions != policy_exemptions:
        raise SophiaUsageError(
            "[ERR] production Chinese quality proxy curated exemptions do not "
            "match policy."
        )
    for index, exemption in enumerate(policy_exemptions):
        label = f"chinese_quality_proxy_exemption_{index:03d}"
        evidence_item = artifacts.get(label)
        if not isinstance(evidence_item, dict):
            raise SophiaUsageError(
                "[ERR] production dataset is missing curated Chinese quality "
                f"evidence: {label}."
            )
        evidence_path = _resolve_relative_file(
            root=root,
            value=evidence_item.get("path"),
            label=f"curated Chinese quality evidence {index}",
        )
        evidence = _load_policy_like_json(
            evidence_path,
            label=f"curated Chinese quality evidence {index}",
        )
        evidence_source = evidence.get("source")
        gate_results = evidence.get("gate_results")
        if (
            str(evidence_item.get("sha256") or "")
            != exemption["evidence_report_sha256"]
            or _sha256_file(evidence_path) != exemption["evidence_report_sha256"]
            or str(evidence.get("schema") or "")
            != "sophia_curated_chinese_source_proxy_exemption_v1"
            or str(evidence.get("status") or "") != "pass"
            or not isinstance(evidence_source, dict)
            or str(evidence_source.get("repo_id") or "").casefold()
            != exemption["repo_id"]
            or str(evidence_source.get("revision") or "").lower()
            != exemption["revision"]
            or not isinstance(gate_results, dict)
            or not gate_results
            or not all(value is True for value in gate_results.values())
        ):
            raise SophiaUsageError(
                "[ERR] production curated Chinese quality evidence is invalid: "
                f"{label}."
            )


def validate_pretrain_data_admission(
    *,
    data_path: str,
    policy_path: str | Path = DEFAULT_ADMISSION_POLICY,
) -> None:
    """Reject legacy corpora at the formal training entrypoint."""

    policy_file = Path(policy_path).expanduser().resolve()
    policy = _load_policy(policy_file)
    forbidden_values = policy.get("forbidden_input_roots")
    if not isinstance(forbidden_values, list) or not forbidden_values:
        raise SophiaUsageError(
            "[ERR] pretrain data admission policy has no forbidden_input_roots."
        )

    forbidden_roots: list[Path] = []
    for value in forbidden_values:
        root = Path(str(value)).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        forbidden_roots.append(root.resolve())

    candidates = (("data_path", data_path),)
    for label, value in candidates:
        raw = str(value or "").strip()
        if label == "data_path" and not raw:
            raise SophiaUsageError("[ERR] data_path is empty.")
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        for forbidden in forbidden_roots:
            if _is_within(candidate, forbidden):
                raise SophiaUsageError(
                    "[ERR] legacy pretraining data is forbidden for the new "
                    "random-initialization model.\n"
                    f"field={label}\n"
                    f"path={candidate}\n"
                    f"forbidden_root={forbidden}\n"
                    "Build fresh shards from the pinned source inventory and new "
                    "tokenizer; do not reuse old corpora or token shards."
                )


def validate_pretrain_dataset_manifest(
    *,
    data_path: str,
    tokenizer_path: str = "",
    policy_path: str | Path = DEFAULT_ADMISSION_POLICY,
) -> None:
    """Validate the self-contained production dataset before release training."""
    root = _dataset_root(data_path)
    dataset_path = root / "dataset_manifest.json"
    marker_path = root / DATASET_MARKER
    if not dataset_path.is_file() or not marker_path.is_file():
        raise SophiaUsageError(
            "[ERR] production pretraining dataset manifest is missing.\n"
            f"expected={dataset_path}\n"
            "Build shards with: ml-tool pretrain shard-build"
        )
    dataset = _load_policy_like_json(dataset_path, label="production dataset manifest")
    marker = _load_policy_like_json(marker_path, label="production dataset marker")
    if (
        str(dataset.get("schema")) != DATASET_SCHEMA
        or str(dataset.get("status")) != "complete"
        or str(marker.get("schema")) != DATASET_SCHEMA
        or str(marker.get("status")) != "complete"
    ):
        raise SophiaUsageError("[ERR] production pretraining dataset is incomplete.")
    expected_dataset_sha256 = str(marker.get("dataset_manifest_sha256") or "")
    if _sha256_file(dataset_path) != expected_dataset_sha256:
        raise SophiaUsageError(
            "[ERR] production dataset manifest fingerprint mismatch."
        )

    policy = _load_policy(Path(policy_path).expanduser().resolve())
    tokenizer_dir = Path(
        str(tokenizer_path or (REPO_ROOT / "ml/modeling/text"))
    ).expanduser().resolve()
    tokenizer_sha1 = compute_tokenizer_bundle_sha1(str(tokenizer_dir))
    expected_tokenizer = str(policy.get("required_tokenizer_bundle_sha1") or "")
    recorded_tokenizers = {
        expected_tokenizer,
        str(dataset.get("tokenizer_bundle_sha1") or ""),
        str(marker.get("tokenizer_bundle_sha1") or ""),
    }
    if recorded_tokenizers != {tokenizer_sha1}:
        raise SophiaUsageError(
            "[ERR] production dataset tokenizer lineage mismatch.\n"
            f"computed={tokenizer_sha1}\nrecorded={sorted(recorded_tokenizers)}"
        )

    lineage = dataset.get("lineage")
    if not isinstance(lineage, dict):
        raise SophiaUsageError("[ERR] production dataset lineage is missing.")
    required_fingerprints = {
        "cleaning_report_sha256",
        "profile_sha256",
        "mix_plan_sha256",
        "mix_policy_sha256",
        "admission_policy_sha256",
    }
    if set(lineage) != required_fingerprints or marker.get("lineage") != lineage:
        raise SophiaUsageError("[ERR] production dataset lineage fingerprints are incomplete.")
    for label, value in lineage.items():
        fingerprint = str(value or "")
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise SophiaUsageError(
                f"[ERR] invalid production dataset lineage fingerprint: {label}"
            )
    policy_sha256 = _sha256_file(Path(policy_path).expanduser().resolve())
    if str(lineage.get("admission_policy_sha256") or "") != policy_sha256:
        raise SophiaUsageError(
            "[ERR] production dataset was built under a different admission policy."
        )
    artifacts = dataset.get("lineage_artifacts")
    required_artifacts = {
        "admission_policy",
        "cleaning_report",
        "token_profile",
        "resolved_mix_plan",
        "mix_policy",
    }
    if not isinstance(artifacts, dict) or not required_artifacts.issubset(artifacts):
        raise SophiaUsageError("[ERR] production dataset lineage artifacts are incomplete.")
    for label, item in artifacts.items():
        if not isinstance(item, dict):
            raise SophiaUsageError(f"[ERR] invalid lineage artifact: {label}")
        artifact_path = _resolve_relative_file(
            root=root,
            value=item.get("path"),
            label=f"lineage artifact {label}",
        )
        if (
            int(artifact_path.stat().st_size) != int(item.get("bytes") or 0)
            or _sha256_file(artifact_path) != str(item.get("sha256") or "")
        ):
            raise SophiaUsageError(
                f"[ERR] production dataset lineage artifact fingerprint mismatch: {label}"
            )
    artifact_to_fingerprint = {
        "admission_policy": "admission_policy_sha256",
        "cleaning_report": "cleaning_report_sha256",
        "token_profile": "profile_sha256",
        "resolved_mix_plan": "mix_plan_sha256",
        "mix_policy": "mix_policy_sha256",
    }
    for artifact_label, fingerprint_label in artifact_to_fingerprint.items():
        if str(artifacts[artifact_label].get("sha256") or "") != str(
            lineage.get(fingerprint_label) or ""
        ):
            raise SophiaUsageError(
                f"[ERR] production dataset lineage binding mismatch: {artifact_label}"
            )
    _validate_chinese_quality_proxy_lineage(
        root=root,
        policy=policy,
        artifacts=artifacts,
    )

    splits = dataset.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise SophiaUsageError("[ERR] production dataset must contain train/val/test splits.")
    for split in ("train", "val", "test"):
        info = splits.get(split)
        if not isinstance(info, dict):
            raise SophiaUsageError(f"[ERR] invalid production dataset split: {split}")
        manifest_path = _resolve_relative_file(
            root=root,
            value=info.get("manifest"),
            label=f"{split} manifest",
        )
        expected_manifest_path = (root / split / "manifest.json").resolve()
        if manifest_path != expected_manifest_path:
            raise SophiaUsageError(
                f"[ERR] production dataset split manifest path mismatch: {split}"
            )
        if sha1_file(str(manifest_path)) != str(info.get("manifest_sha1") or ""):
            raise SophiaUsageError(
                f"[ERR] production dataset split manifest fingerprint mismatch: {split}"
            )
        try:
            manifest = load_manifest(str(manifest_path))
        except (OSError, ValueError) as exc:
            raise SophiaUsageError(str(exc)) from exc
        if (
            manifest.tokenizer_sha1 != tokenizer_sha1
            or manifest.total_tokens != int(info.get("tokens") or 0)
        ):
            raise SophiaUsageError(
                f"[ERR] production dataset split metadata mismatch: {split}"
            )
        shard_inventory = info.get("shards")
        if not isinstance(shard_inventory, list) or len(shard_inventory) != len(
            manifest.shards
        ):
            raise SophiaUsageError(
                f"[ERR] production dataset shard inventory mismatch: {split}"
            )
        for manifest_shard, shard_info in zip(
            manifest.shards, shard_inventory, strict=True
        ):
            if not isinstance(shard_info, dict) or str(
                shard_info.get("path") or ""
            ) != str(manifest_shard.path):
                raise SophiaUsageError(
                    f"[ERR] production dataset shard path mismatch: {split}"
                )
            shard_path = _resolve_relative_file(
                root=manifest_path.parent,
                value=manifest_shard.path,
                label=f"{split} shard",
            )
            if (
                int(manifest_shard.tokens) != int(shard_info.get("tokens") or 0)
                or int(shard_path.stat().st_size) != int(shard_info.get("bytes") or 0)
                or _sha256_file(shard_path) != str(shard_info.get("sha256") or "")
            ):
                raise SophiaUsageError(
                    f"[ERR] production dataset shard fingerprint mismatch: {shard_path}"
                )


def _load_policy_like_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SophiaUsageError(f"[ERR] unable to load {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SophiaUsageError(f"[ERR] invalid {label}: {path}")
    return payload


__all__ = [
    "DEFAULT_ADMISSION_POLICY",
    "DATASET_MARKER",
    "DATASET_SCHEMA",
    "POLICY_SCHEMA",
    "validate_pretrain_data_admission",
    "validate_pretrain_dataset_manifest",
]
