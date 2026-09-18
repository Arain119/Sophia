from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

from ml.data.pretrain_filters import repeated_sentence_excess


POLICY_KEY = "post_clean_document_admission"
_DOCUMENT_SHA256_RE = re.compile(rb"[0-9a-f]{64}\Z")
_REASON_RE = re.compile(rb"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class RepeatedSentenceRule:
    min_sentence_chars: int
    repeats: int
    minimum_repeated_chars: int
    minimum_repeated_fraction: float


@dataclass(frozen=True)
class DocumentExclusionManifest:
    path: str
    sha256: str
    entries: int
    sources: frozenset[str]
    digests: bytes
    reason_ids: bytes
    reasons: tuple[str, ...]
    reason_counts: Mapping[str, int]

    def applies_to_source(self, source: object) -> bool:
        return str(source or "") in self.sources

    def reason_for(self, document_sha256: object) -> str:
        raw = str(document_sha256 or "").encode("ascii", errors="ignore")
        if _DOCUMENT_SHA256_RE.fullmatch(raw) is None:
            return ""
        target = bytes.fromhex(raw.decode("ascii"))
        low = 0
        high = self.entries
        while low < high:
            middle = (low + high) // 2
            offset = middle * 32
            candidate = self.digests[offset : offset + 32]
            if candidate < target:
                low = middle + 1
            else:
                high = middle
        if low >= self.entries:
            return ""
        offset = low * 32
        if self.digests[offset : offset + 32] != target:
            return ""
        return self.reasons[self.reason_ids[low]]

    def to_report(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "entries": self.entries,
            "sources": sorted(self.sources),
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


@dataclass(frozen=True)
class PostCleanDocumentAdmission:
    exclude_legacy_exact_overlap: bool
    exclude_legacy_near_overlap: bool
    repeated_sentence_rules: Mapping[str, RepeatedSentenceRule]
    document_exclusion_manifest: DocumentExclusionManifest | None = None

    @property
    def required_columns(self) -> set[str]:
        columns: set[str] = set()
        if self.exclude_legacy_exact_overlap:
            columns.add("legacy_exact_overlap")
        if self.exclude_legacy_near_overlap:
            columns.add("legacy_near_overlap")
        if self.document_exclusion_manifest is not None:
            columns.add("document_sha256")
        return columns

    def exclusion_reason(self, row: Mapping[str, Any]) -> str:
        if self.exclude_legacy_exact_overlap and bool(row.get("legacy_exact_overlap")):
            return "legacy_exact_overlap"
        if self.exclude_legacy_near_overlap and bool(row.get("legacy_near_overlap")):
            return "legacy_near_overlap"
        source = str(row.get("source") or "")
        if (
            self.document_exclusion_manifest is not None
            and self.document_exclusion_manifest.applies_to_source(source)
        ):
            reason = self.document_exclusion_manifest.reason_for(
                row.get("document_sha256")
            )
            if reason:
                return reason
        rule = self.repeated_sentence_rules.get(source)
        if rule is None:
            return ""
        repeated_characters, repeated_fraction = repeated_sentence_excess(
            str(row.get("text") or ""),
            min_sentence_chars=rule.min_sentence_chars,
            repeats=rule.repeats,
        )
        if (
            repeated_characters >= rule.minimum_repeated_chars
            and repeated_fraction >= rule.minimum_repeated_fraction
        ):
            return "severe_internal_sentence_repetition"
        return ""

    def to_report(self) -> dict[str, Any]:
        return {
            "exclude_legacy_exact_overlap": self.exclude_legacy_exact_overlap,
            "exclude_legacy_near_overlap": self.exclude_legacy_near_overlap,
            "document_exclusion_manifest": (
                self.document_exclusion_manifest.to_report()
                if self.document_exclusion_manifest is not None
                else None
            ),
            "repeated_sentence_rules": {
                source: {
                    "min_sentence_chars": rule.min_sentence_chars,
                    "repeats": rule.repeats,
                    "minimum_repeated_chars": rule.minimum_repeated_chars,
                    "minimum_repeated_fraction": rule.minimum_repeated_fraction,
                }
                for source, rule in sorted(self.repeated_sentence_rules.items())
            },
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_document_exclusion_manifest(
    raw: object,
    *,
    policy_path: str | Path | None,
) -> DocumentExclusionManifest | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{POLICY_KEY}.document_exclusion_manifest must be an object")
    allowed = {"path", "sha256", "entries", "sources"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown document exclusion manifest keys: {unknown}")
    configured_path = str(raw.get("path") or "").strip()
    expected_sha256 = str(raw.get("sha256") or "").strip()
    expected_entries = int(raw.get("entries") or 0)
    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("document exclusion manifest sources must be a list")
    sources = frozenset(str(value).strip() for value in raw_sources if str(value).strip())
    if not sources or len(sources) != len(raw_sources):
        raise ValueError("document exclusion manifest sources must be unique and non-empty")
    if not configured_path or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("document exclusion manifest path and sha256 are required")
    if expected_entries <= 0:
        raise ValueError("document exclusion manifest entries must be positive")
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        if policy_path is None:
            raise ValueError(
                "policy_path is required for a relative document exclusion manifest"
            )
        path = Path(policy_path).expanduser().resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"document exclusion manifest does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "document exclusion manifest sha256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )

    digests = bytearray()
    reason_ids = bytearray()
    reasons: list[str] = []
    reason_indexes: dict[str, int] = {}
    reason_counts: Counter[str] = Counter()
    previous_digest = b""
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith(b"\n") or line.endswith(b"\r\n"):
                raise ValueError(
                    f"document exclusion manifest must use LF-terminated lines: {line_number}"
                )
            fields = line[:-1].split(b"\t")
            if len(fields) != 2:
                raise ValueError(
                    f"invalid document exclusion manifest row: {line_number}"
                )
            digest_text, reason_text = fields
            if _DOCUMENT_SHA256_RE.fullmatch(digest_text) is None:
                raise ValueError(
                    f"invalid document sha256 in exclusion manifest: {line_number}"
                )
            if _REASON_RE.fullmatch(reason_text) is None:
                raise ValueError(
                    f"invalid reason in document exclusion manifest: {line_number}"
                )
            digest = bytes.fromhex(digest_text.decode("ascii"))
            if previous_digest and digest <= previous_digest:
                raise ValueError(
                    "document exclusion manifest hashes must be strictly sorted and unique"
                )
            previous_digest = digest
            reason = reason_text.decode("ascii")
            reason_index = reason_indexes.get(reason)
            if reason_index is None:
                reason_index = len(reasons)
                if reason_index >= 256:
                    raise ValueError("document exclusion manifest has too many reasons")
                reason_indexes[reason] = reason_index
                reasons.append(reason)
            digests.extend(digest)
            reason_ids.append(reason_index)
            reason_counts[reason] += 1
    actual_entries = len(reason_ids)
    if actual_entries != expected_entries:
        raise ValueError(
            "document exclusion manifest entry count mismatch: "
            f"expected={expected_entries} actual={actual_entries}"
        )
    return DocumentExclusionManifest(
        path=str(path),
        sha256=actual_sha256,
        entries=actual_entries,
        sources=sources,
        digests=bytes(digests),
        reason_ids=bytes(reason_ids),
        reasons=tuple(reasons),
        reason_counts=dict(reason_counts),
    )


def load_post_clean_document_admission(
    policy: Mapping[str, Any],
    *,
    policy_path: str | Path | None = None,
) -> PostCleanDocumentAdmission:
    raw = policy.get(POLICY_KEY)
    if raw is None:
        return PostCleanDocumentAdmission(False, False, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{POLICY_KEY} must be an object")
    allowed = {
        "exclude_legacy_exact_overlap",
        "exclude_legacy_near_overlap",
        "document_exclusion_manifest",
        "repeated_sentence_rules",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {POLICY_KEY} keys: {unknown}")
    raw_rules = raw.get("repeated_sentence_rules", {})
    if not isinstance(raw_rules, dict):
        raise ValueError(f"{POLICY_KEY}.repeated_sentence_rules must be an object")
    rules: dict[str, RepeatedSentenceRule] = {}
    rule_keys = {
        "min_sentence_chars",
        "repeats",
        "minimum_repeated_chars",
        "minimum_repeated_fraction",
    }
    for source, value in raw_rules.items():
        source_name = str(source).strip()
        if not source_name or not isinstance(value, dict):
            raise ValueError("repeated sentence rule source and value must be valid")
        rule_unknown = sorted(set(value) - rule_keys)
        if rule_unknown:
            raise ValueError(
                f"unknown repeated sentence rule keys for {source_name}: {rule_unknown}"
            )
        rule = RepeatedSentenceRule(
            min_sentence_chars=int(value.get("min_sentence_chars", 20)),
            repeats=int(value.get("repeats", 2)),
            minimum_repeated_chars=int(value.get("minimum_repeated_chars", 240)),
            minimum_repeated_fraction=float(
                value.get("minimum_repeated_fraction", 0.20)
            ),
        )
        if (
            rule.min_sentence_chars <= 0
            or rule.repeats < 2
            or rule.minimum_repeated_chars <= 0
            or not 0.0 < rule.minimum_repeated_fraction <= 1.0
        ):
            raise ValueError(f"invalid repeated sentence rule for {source_name}")
        rules[source_name] = rule
    return PostCleanDocumentAdmission(
        exclude_legacy_exact_overlap=bool(
            raw.get("exclude_legacy_exact_overlap", False)
        ),
        exclude_legacy_near_overlap=bool(raw.get("exclude_legacy_near_overlap", False)),
        repeated_sentence_rules=rules,
        document_exclusion_manifest=_load_document_exclusion_manifest(
            raw.get("document_exclusion_manifest"),
            policy_path=policy_path,
        ),
    )
