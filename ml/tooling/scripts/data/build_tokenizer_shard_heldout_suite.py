from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from ml.core.common.io import write_json_atomic
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1


REPORT_SCHEMA = "sophia_tokenizer_shard_heldout_suite_build_v1"
SUITE_SCHEMA = "sophia_tokenizer_quality_suite_v1"
CORPUS_DOMAINS = ("native_chinese", "english", "code", "math")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer_quotas(total: int, weights: dict[str, int]) -> dict[str, int]:
    weight_sum = sum(weights.values())
    if total <= 0 or weight_sum <= 0 or any(value <= 0 for value in weights.values()):
        raise ValueError("sample total and source weights must be positive")
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    order = sorted(weights, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[: total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def _source_shards(
    *,
    shard_root: Path,
    manifest: dict[str, Any],
    build_report: dict[str, Any],
) -> tuple[dict[str, list[tuple[Path, int]]], dict[str, int]]:
    manifest_shards = manifest.get("shards")
    report_shards = build_report.get("shards")
    tokens_by = build_report.get("tokens_by")
    source_tokens = tokens_by.get("source") if isinstance(tokens_by, dict) else None
    if not isinstance(manifest_shards, list) or not isinstance(report_shards, list):
        raise ValueError("manifest/build report is missing shards")
    if manifest_shards != [
        {"path": row.get("path"), "tokens": row.get("tokens")}
        for row in report_shards
    ]:
        raise ValueError("manifest and build-report shard order/tokens differ")
    if not isinstance(source_tokens, dict) or not source_tokens:
        raise ValueError("build report is missing tokens_by.source")

    rows: dict[str, list[tuple[Path, int]]] = {}
    cursor = 0
    for raw_source, raw_target in source_tokens.items():
        source = str(raw_source)
        target = int(raw_target)
        selected: list[tuple[Path, int]] = []
        selected_tokens = 0
        while cursor < len(manifest_shards) and selected_tokens < target:
            shard = manifest_shards[cursor]
            cursor += 1
            if not isinstance(shard, dict):
                raise ValueError("invalid shard row")
            tokens = int(shard.get("tokens", 0) or 0)
            path = shard_root / str(shard.get("path") or "")
            if tokens <= 0 or not path.is_file():
                raise ValueError(f"missing or invalid shard: {path}")
            expected_bytes = tokens * np.dtype("<i4").itemsize
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"shard size mismatch: {path}")
            selected.append((path, tokens))
            selected_tokens += tokens
        if selected_tokens != target:
            raise ValueError(
                f"source/shard token boundary mismatch: {source} "
                f"expected={target} actual={selected_tokens}"
            )
        rows[source] = selected
    if cursor != len(manifest_shards):
        raise ValueError("unassigned shards remain after source reconstruction")
    return rows, {str(key): int(value) for key, value in source_tokens.items()}


def _validate_domains(
    *, source_domains: dict[str, Any], available_sources: set[str]
) -> dict[str, list[str]]:
    raw_domains = source_domains.get("domains")
    if not isinstance(raw_domains, dict):
        raise ValueError("source-domain config is missing domains")
    domains: dict[str, list[str]] = {}
    assigned: list[str] = []
    for domain in CORPUS_DOMAINS:
        raw_sources = raw_domains.get(domain)
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"source-domain config is missing {domain}")
        sources = [str(value) for value in raw_sources]
        domains[domain] = sources
        assigned.extend(sources)
    duplicates = sorted(source for source, count in Counter(assigned).items() if count != 1)
    if duplicates:
        raise ValueError(f"sources assigned to multiple domains: {duplicates}")
    if set(assigned) != available_sources:
        raise ValueError(
            "source-domain config must partition every held-out source: "
            f"missing={sorted(available_sources - set(assigned))} "
            f"unknown={sorted(set(assigned) - available_sources)}"
        )
    return domains


def _sample_document(
    *,
    source: str,
    sample_index: int,
    shards: list[tuple[Path, int]],
    eos_token_id: int,
    max_scan_tokens: int,
) -> list[int]:
    cumulative: list[int] = []
    total = 0
    for _path, tokens in shards:
        total += int(tokens)
        cumulative.append(total)
    digest = hashlib.sha256(f"{source}\0{sample_index}".encode()).digest()
    global_offset = int.from_bytes(digest[:8], "big") % total
    shard_index = bisect_right(cumulative, global_offset)
    previous = 0 if shard_index == 0 else cumulative[shard_index - 1]
    path, tokens = shards[shard_index]
    local_offset = global_offset - previous
    values = np.memmap(path, mode="r", dtype="<i4", shape=(tokens,))
    left = max(local_offset - int(max_scan_tokens), 0)
    right = min(local_offset + int(max_scan_tokens) + 1, tokens)
    window = np.asarray(values[left:right])
    relative = local_offset - left
    before = np.flatnonzero(window[:relative] == int(eos_token_id))
    after = np.flatnonzero(window[relative:] == int(eos_token_id))
    start = left + (int(before[-1]) + 1 if before.size else 0)
    end = left + relative + (int(after[0]) if after.size else 0)
    if not after.size or end <= start:
        return []
    return [int(value) for value in values[start:end]]


def build_suite(
    *,
    shard_root: Path,
    tokenizer_dir: Path,
    source_domains_path: Path,
    boundary_suite_path: Path,
    output_path: Path,
    report_path: Path,
    samples_per_domain: int,
    max_scan_tokens: int,
    max_text_chars: int,
) -> dict[str, Any]:
    manifest_path = shard_root / "manifest.json"
    build_report_path = shard_root / "build_report.json"
    manifest = _load_json(manifest_path)
    build_report = _load_json(build_report_path)
    bundle_sha1 = compute_tokenizer_bundle_sha1(str(tokenizer_dir))
    recorded_hashes = {
        str(manifest.get("tokenizer_sha1") or ""),
        str(build_report.get("tokenizer_bundle_sha1") or ""),
    }
    if recorded_hashes != {bundle_sha1}:
        raise ValueError(
            "held-out shards and tokenizer bundle do not match: "
            f"recorded={sorted(recorded_hashes)} actual={bundle_sha1}"
        )
    source_shards, source_tokens = _source_shards(
        shard_root=shard_root,
        manifest=manifest,
        build_report=build_report,
    )
    domain_sources = _validate_domains(
        source_domains=_load_json(source_domains_path),
        available_sources=set(source_shards),
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    eos_token_id = int(manifest.get("eos_token_id", -1))
    if eos_token_id < 0:
        raise ValueError("held-out shard manifest has no EOS token id")

    domains: dict[str, list[str]] = {}
    selected_by_source: Counter[str] = Counter()
    for domain in CORPUS_DOMAINS:
        sources = domain_sources[domain]
        quotas = _integer_quotas(
            int(samples_per_domain),
            {source: source_tokens[source] for source in sources},
        )
        texts: list[str] = []
        seen: set[str] = set()
        for source in sources:
            quota = quotas[source]
            attempts = 0
            accepted = 0
            while accepted < quota and attempts < max(quota * 100, 100):
                token_ids = _sample_document(
                    source=source,
                    sample_index=attempts,
                    shards=source_shards[source],
                    eos_token_id=eos_token_id,
                    max_scan_tokens=int(max_scan_tokens),
                )
                attempts += 1
                if not token_ids:
                    continue
                text = tokenizer.decode(token_ids, skip_special_tokens=False).strip()
                text = text[: int(max_text_chars)].strip()
                fingerprint = hashlib.sha256(text.encode()).hexdigest()
                if len(text) < 64 or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                texts.append(text)
                selected_by_source[source] += 1
                accepted += 1
            if accepted != quota:
                raise RuntimeError(
                    f"insufficient unique held-out documents: source={source} "
                    f"quota={quota} selected={accepted} attempts={attempts}"
                )
        if len(texts) != int(samples_per_domain):
            raise RuntimeError(f"held-out sample count drifted for domain {domain}")
        domains[domain] = texts

    boundary = _load_json(boundary_suite_path).get("domains")
    if not isinstance(boundary, dict):
        raise ValueError("boundary suite has no domains")
    for domain in ("tool_json", "unicode_robustness"):
        texts = boundary.get(domain)
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"boundary suite is missing {domain}")
        domains[domain] = [str(text) for text in texts]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(output_path), {"schema": SUITE_SCHEMA, "domains": domains})
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "suite_path": str(output_path.resolve()),
        "suite_sha256": _sha256(output_path),
        "sample_count": sum(len(values) for values in domains.values()),
        "samples_per_corpus_domain": int(samples_per_domain),
        "selected_by_source": dict(sorted(selected_by_source.items())),
        "tokenizer_bundle_sha1": bundle_sha1,
        "inputs": {
            "manifest_sha256": _sha256(manifest_path),
            "build_report_sha256": _sha256(build_report_path),
            "source_domains_sha256": _sha256(source_domains_path),
            "boundary_suite_sha256": _sha256(boundary_suite_path),
        },
    }
    write_json_atomic(str(report_path), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic tokenizer suite from held-out token shards."
    )
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source-domains", required=True)
    parser.add_argument("--boundary-suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--samples-per-domain", type=int, default=250)
    parser.add_argument("--max-scan-tokens", type=int, default=4096)
    parser.add_argument("--max-text-chars", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        int(args.samples_per_domain),
        int(args.max_scan_tokens),
        int(args.max_text_chars),
    ) <= 0:
        raise SystemExit("sample count and scan/text limits must be positive")
    report = build_suite(
        shard_root=Path(args.shard_root).resolve(),
        tokenizer_dir=Path(args.tokenizer).resolve(),
        source_domains_path=Path(args.source_domains).resolve(),
        boundary_suite_path=Path(args.boundary_suite).resolve(),
        output_path=Path(args.output).resolve(),
        report_path=Path(args.report).resolve(),
        samples_per_domain=int(args.samples_per_domain),
        max_scan_tokens=int(args.max_scan_tokens),
        max_text_chars=int(args.max_text_chars),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
