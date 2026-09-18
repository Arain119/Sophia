from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.data.pretrain_document_admission import load_post_clean_document_admission


REPORT_SCHEMA = "sophia_tokenizer_heldout_suite_build_v1"
SUITE_SCHEMA = "sophia_tokenizer_quality_suite_v1"
CORPUS_DOMAINS = ("native_chinese", "english", "code", "math")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer_quotas(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    weight_sum = sum(int(value) for value in weights.values())
    if total <= 0 or weight_sum <= 0 or any(int(value) <= 0 for value in weights.values()):
        raise ValueError("sample total and source weights must be positive")
    exact = {source: total * int(value) / float(weight_sum) for source, value in weights.items()}
    quotas = {source: int(math.floor(value)) for source, value in exact.items()}
    order = sorted(weights, key=lambda source: (-(exact[source] - quotas[source]), source))
    for source in order[: int(total) - sum(quotas.values())]:
        quotas[source] += 1
    return dict(sorted(quotas.items()))


def _domain_sources(
    *, plan: dict[str, Any], mix_policy: dict[str, Any]
) -> dict[str, list[str]]:
    source_quotas = plan.get("source_train_quotas")
    constraints = mix_policy.get("constraints")
    groups = constraints.get("source_groups") if isinstance(constraints, dict) else None
    if not isinstance(source_quotas, dict) or not isinstance(groups, dict):
        raise ValueError("mix inputs do not contain source quotas and source groups")
    all_sources = set(str(source) for source in source_quotas)
    chinese = set(str(source) for source in groups.get("chinese", []))
    code = set(str(source) for source in groups.get("code", []))
    math_sources = set(str(source) for source in groups.get("math_stem", []))
    assigned = chinese | code | math_sources
    if not chinese or not code or not math_sources or not assigned <= all_sources:
        raise ValueError("tokenizer held-out domain groups are incomplete or unknown")
    return {
        "native_chinese": sorted(chinese),
        "english": sorted(all_sources - assigned),
        "code": sorted(code),
        "math": sorted(math_sources),
    }


def _stable_source_sample(
    *,
    clean_root: Path,
    source: str,
    quota: int,
    admission: Any,
    max_text_chars: int,
    batch_size: int,
) -> tuple[list[str], dict[str, Any]]:
    if quota <= 0:
        return [], {"quota": 0, "scanned": 0, "excluded": 0, "selected": 0}
    files = [
        path
        for split in ("val", "test")
        for path in sorted((clean_root / source / split).rglob("*.parquet"))
    ]
    if not files:
        raise FileNotFoundError(f"no held-out parquet files for source: {source}")
    heap: list[tuple[int, str]] = []
    counters: Counter[str] = Counter()
    required = {"text", "source"} | admission.required_columns
    for path in files:
        parquet = pq.ParquetFile(path)
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"held-out input is missing columns {missing}: {path}")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_size):
            for row in batch.to_pylist():
                counters["scanned"] += 1
                if str(row.get("source") or "") != source:
                    raise ValueError(f"source mismatch in held-out file: {path}")
                reason = admission.exclusion_reason(row)
                if reason:
                    counters["excluded"] += 1
                    counters[f"excluded:{reason}"] += 1
                    continue
                text = str(row.get("text") or "").strip()
                if len(text) < 64:
                    counters["too_short"] += 1
                    continue
                text = text[:max_text_chars].strip()
                score = int.from_bytes(
                    hashlib.sha256(f"{source}\0{text}".encode()).digest(), "big"
                )
                item = (-score, text)
                if len(heap) < quota:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    if len(heap) != quota:
        raise RuntimeError(
            f"insufficient held-out documents: source={source} quota={quota} selected={len(heap)}"
        )
    selected = [text for _negative_score, text in sorted(heap, reverse=True)]
    return selected, {
        "quota": int(quota),
        "files": len(files),
        "scanned": int(counters["scanned"]),
        "excluded": int(counters["excluded"]),
        "excluded_by": {
            key.removeprefix("excluded:"): int(value)
            for key, value in sorted(counters.items())
            if key.startswith("excluded:")
        },
        "too_short": int(counters["too_short"]),
        "selected": len(selected),
    }


def build_suite(
    *,
    clean_root: Path,
    resolved_mix_plan_path: Path,
    mix_policy_path: Path,
    admission_policy_path: Path,
    boundary_suite_path: Path,
    output_path: Path,
    report_path: Path,
    samples_per_domain: int,
    max_text_chars: int,
    batch_size: int,
) -> dict[str, Any]:
    plan = _load_json(resolved_mix_plan_path)
    if plan.get("status") != "ready":
        raise ValueError("held-out tokenizer suite requires a ready mix plan")
    mix_policy = _load_json(mix_policy_path)
    admission = load_post_clean_document_admission(
        _load_json(admission_policy_path),
        policy_path=admission_policy_path,
    )
    source_weights = {
        str(source): int(value) for source, value in plan["source_train_quotas"].items()
    }
    domain_sources = _domain_sources(plan=plan, mix_policy=mix_policy)
    domains: dict[str, list[str]] = {}
    source_reports: dict[str, Any] = {}
    domain_quotas: dict[str, dict[str, int]] = {}
    for domain in CORPUS_DOMAINS:
        members = domain_sources[domain]
        quotas = _integer_quotas(
            int(samples_per_domain), {source: source_weights[source] for source in members}
        )
        domain_quotas[domain] = quotas
        texts: list[str] = []
        for source, quota in quotas.items():
            sampled, source_report = _stable_source_sample(
                clean_root=clean_root,
                source=source,
                quota=quota,
                admission=admission,
                max_text_chars=int(max_text_chars),
                batch_size=int(batch_size),
            )
            texts.extend(sampled)
            source_reports[source] = source_report
        if len(texts) != int(samples_per_domain):
            raise RuntimeError(f"held-out domain sample count drifted: {domain}")
        domains[domain] = texts
    boundary = _load_json(boundary_suite_path).get("domains")
    if not isinstance(boundary, dict):
        raise ValueError("boundary tokenizer suite has no domains")
    for domain in ("tool_json", "unicode_robustness"):
        raw_texts = boundary.get(domain)
        if not isinstance(raw_texts, list) or not raw_texts:
            raise ValueError(f"boundary tokenizer suite is missing {domain}")
        domains[domain] = [str(text) for text in raw_texts]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(output_path), {"schema": SUITE_SCHEMA, "domains": domains})
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "suite_path": str(output_path),
        "suite_sha256": _sha256(output_path),
        "samples_per_corpus_domain": int(samples_per_domain),
        "max_text_chars": int(max_text_chars),
        "domain_source_quotas": domain_quotas,
        "source_reports": source_reports,
        "post_clean_document_admission": admission.to_report(),
        "inputs": {
            "resolved_mix_plan_sha256": _sha256(resolved_mix_plan_path),
            "mix_policy_sha256": _sha256(mix_policy_path),
            "admission_policy_sha256": _sha256(admission_policy_path),
            "boundary_suite_sha256": _sha256(boundary_suite_path),
        },
    }
    write_json_atomic(str(report_path), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a mix-aligned held-out tokenizer suite.")
    parser.add_argument("--clean_root", required=True)
    parser.add_argument("--resolved_mix_plan", required=True)
    parser.add_argument("--mix_policy", required=True)
    parser.add_argument("--admission_policy", required=True)
    parser.add_argument("--boundary_suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--samples_per_domain", type=int, default=2000)
    parser.add_argument("--max_text_chars", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(int(args.samples_per_domain), int(args.max_text_chars), int(args.batch_size)) <= 0:
        raise SystemExit("sample count, max text chars, and batch size must be positive")
    report = build_suite(
        clean_root=Path(args.clean_root).resolve(),
        resolved_mix_plan_path=Path(args.resolved_mix_plan).resolve(),
        mix_policy_path=Path(args.mix_policy).resolve(),
        admission_policy_path=Path(args.admission_policy).resolve(),
        boundary_suite_path=Path(args.boundary_suite).resolve(),
        output_path=Path(args.output).resolve(),
        report_path=Path(args.report).resolve(),
        samples_per_domain=int(args.samples_per_domain),
        max_text_chars=int(args.max_text_chars),
        batch_size=int(args.batch_size),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
