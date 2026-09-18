from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import statistics
import time
from typing import Any

import tokenizers
from tokenizers import Tokenizer

from ml.core.common.io import write_json_atomic
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.tooling.scripts.data.train_tokenizer import CORE_SPECIAL_TOKENS


REPORT_SCHEMA = "sophia_tokenizer_quality_audit_v1"
_FALLBACK_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "tokenizer_report.json",
    "tokenizer_provenance.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_suite(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("domains"), dict):
        raise ValueError("tokenizer suite must contain a domains object")
    domains: dict[str, list[str]] = {}
    for raw_name, raw_samples in payload["domains"].items():
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError(f"tokenizer suite domain {raw_name!r} must be non-empty")
        samples = [str(sample) for sample in raw_samples]
        if any(not sample for sample in samples):
            raise ValueError(f"tokenizer suite domain {raw_name!r} contains empty text")
        domains[str(raw_name)] = samples
    return domains


def _tokenizer_model_metadata(tokenizer_json: Path) -> dict[str, Any]:
    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f"tokenizer.json has no model object: {tokenizer_json}")
    return {
        "type": str(model.get("type", "")),
        "byte_fallback": bool(model.get("byte_fallback", False)),
        "unk_token": str(model.get("unk_token", "") or ""),
    }


def _tokenizer_config_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"model_max_length": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"tokenizer config must be an object: {path}")
    raw_max_length = payload.get("model_max_length")
    return {
        "model_max_length": int(raw_max_length)
        if isinstance(raw_max_length, (int, float))
        else None
    }


def _sample_stats(
    *,
    tokenizer: Tokenizer,
    text: str,
    unk_token_id: int | None,
) -> dict[str, Any]:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = [int(value) for value in encoding.ids]
    tokens = [str(value) for value in encoding.tokens]
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    token_count = len(ids)
    char_count = len(text)
    byte_count = len(text.encode("utf-8"))
    unk_count = 0 if unk_token_id is None else sum(value == unk_token_id for value in ids)
    fallback_count = sum(bool(_FALLBACK_TOKEN_RE.fullmatch(token)) for token in tokens)
    return {
        "text": text,
        "chars": char_count,
        "utf8_bytes": byte_count,
        "tokens": token_count,
        "tokens_per_char": token_count / float(max(char_count, 1)),
        "chars_per_token": char_count / float(max(token_count, 1)),
        "bytes_per_token": byte_count / float(max(token_count, 1)),
        "unk_tokens": unk_count,
        "byte_fallback_tokens": fallback_count,
        "roundtrip_ok": decoded == text,
        "decoded": decoded if decoded != text else None,
        "ids_preview": ids[:64],
        "tokens_preview": tokens[:32],
    }


def _domain_stats(
    rows: list[dict[str, Any]], *, max_details: int = 0
) -> dict[str, Any]:
    chars = sum(int(row["chars"]) for row in rows)
    byte_count = sum(int(row["utf8_bytes"]) for row in rows)
    token_count = sum(int(row["tokens"]) for row in rows)
    unk_count = sum(int(row["unk_tokens"]) for row in rows)
    fallback_count = sum(int(row["byte_fallback_tokens"]) for row in rows)
    return {
        "samples": len(rows),
        "chars": chars,
        "utf8_bytes": byte_count,
        "tokens": token_count,
        "tokens_per_char": token_count / float(max(chars, 1)),
        "chars_per_token": chars / float(max(token_count, 1)),
        "bytes_per_token": byte_count / float(max(token_count, 1)),
        "unk_tokens": unk_count,
        "unk_token_rate": unk_count / float(max(token_count, 1)),
        "byte_fallback_tokens": fallback_count,
        "byte_fallback_rate": fallback_count / float(max(token_count, 1)),
        "roundtrip_failures": sum(not bool(row["roundtrip_ok"]) for row in rows),
        "details": rows if int(max_details) <= 0 else rows[: int(max_details)],
    }


def _throughput(
    *, tokenizer: Tokenizer, texts: list[str], warmup: int, iterations: int
) -> dict[str, Any]:
    for _ in range(max(int(warmup), 0)):
        tokenizer.encode_batch(texts, add_special_tokens=False)
    durations: list[float] = []
    token_counts: list[int] = []
    for _ in range(max(int(iterations), 1)):
        started = time.perf_counter()
        encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
        durations.append(time.perf_counter() - started)
        token_counts.append(sum(len(row.ids) for row in encoded))
    median_seconds = statistics.median(durations)
    chars = sum(len(text) for text in texts)
    tokens_count = int(statistics.median(token_counts))
    return {
        "texts_per_iteration": len(texts),
        "chars_per_iteration": chars,
        "tokens_per_iteration": tokens_count,
        "warmup_iterations": max(int(warmup), 0),
        "measured_iterations": max(int(iterations), 1),
        "median_seconds": median_seconds,
        "chars_per_second": chars / max(median_seconds, 1e-12),
        "tokens_per_second": tokens_count / max(median_seconds, 1e-12),
    }


def audit_tokenizer(
    *,
    tokenizer_dir: Path,
    suite_path: Path,
    warmup: int,
    iterations: int,
    max_details_per_domain: int = 0,
) -> dict[str, Any]:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tokenizer.json: {tokenizer_path}")
    domains = _load_suite(suite_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_metadata = _tokenizer_model_metadata(tokenizer_path)
    unk_token = str(model_metadata["unk_token"])
    unk_token_id = tokenizer.token_to_id(unk_token) if unk_token else None

    domain_reports: dict[str, Any] = {}
    all_texts: list[str] = []
    for name, texts in domains.items():
        rows = [
            _sample_stats(
                tokenizer=tokenizer,
                text=text,
                unk_token_id=unk_token_id,
            )
            for text in texts
        ]
        domain_reports[name] = _domain_stats(
            rows, max_details=int(max_details_per_domain)
        )
        all_texts.extend(texts)

    missing_special_tokens = [
        token for token in CORE_SPECIAL_TOKENS if tokenizer.token_to_id(token) is None
    ]
    files = {
        name: {
            "present": (tokenizer_dir / name).is_file(),
            "sha256": _sha256(tokenizer_dir / name)
            if (tokenizer_dir / name).is_file()
            else None,
        }
        for name in _BUNDLE_FILES
    }
    total_tokens = sum(int(row["tokens"]) for row in domain_reports.values())
    total_unk = sum(int(row["unk_tokens"]) for row in domain_reports.values())
    total_fallback = sum(
        int(row["byte_fallback_tokens"]) for row in domain_reports.values()
    )
    roundtrip_failures = sum(
        int(row["roundtrip_failures"]) for row in domain_reports.values()
    )
    provenance_present = bool(files["tokenizer_provenance.json"]["present"])
    training_report_present = bool(files["tokenizer_report.json"]["present"])
    return {
        "schema": REPORT_SCHEMA,
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "suite": {
            "path": str(suite_path.resolve()),
            "sha256": _sha256(suite_path),
            "domain_count": len(domains),
            "sample_count": len(all_texts),
        },
        "bundle": {
            "sha1": compute_tokenizer_bundle_sha1(str(tokenizer_dir)),
            "files": files,
        },
        "model": {
            **model_metadata,
            **_tokenizer_config_metadata(tokenizer_dir / "tokenizer_config.json"),
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        },
        "special_tokens": {
            "required": list(CORE_SPECIAL_TOKENS),
            "missing": missing_special_tokens,
            "ids": {
                token: tokenizer.token_to_id(token) for token in CORE_SPECIAL_TOKENS
            },
        },
        "summary": {
            "tokens": total_tokens,
            "unk_tokens": total_unk,
            "unk_token_rate": total_unk / float(max(total_tokens, 1)),
            "byte_fallback_tokens": total_fallback,
            "byte_fallback_rate": total_fallback / float(max(total_tokens, 1)),
            "roundtrip_failures": roundtrip_failures,
            "missing_special_token_count": len(missing_special_tokens),
            "provenance_present": provenance_present,
            "training_report_present": training_report_present,
        },
        "domains": domain_reports,
        "throughput": _throughput(
            tokenizer=tokenizer,
            texts=all_texts,
            warmup=warmup,
            iterations=iterations,
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tokenizers": tokenizers.__version__,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit tokenizer fertility, fallback, roundtrip, protocol, and throughput."
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--max_details_per_domain",
        type=int,
        default=0,
        help="Keep at most this many per-sample previews per domain; 0 keeps all.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        int(args.warmup) < 0
        or int(args.iterations) <= 0
        or int(args.max_details_per_domain) < 0
    ):
        raise SystemExit(
            "warmup and max_details_per_domain must be >= 0; iterations must be > 0"
        )
    report = audit_tokenizer(
        tokenizer_dir=Path(args.tokenizer),
        suite_path=Path(args.suite),
        warmup=int(args.warmup),
        iterations=int(args.iterations),
        max_details_per_domain=int(args.max_details_per_domain),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(output), report)
    print(
        f"wrote tokenizer audit for {report['suite']['sample_count']} samples "
        f"to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
