#!/usr/bin/env python
"""
Visual audit for token-shard pretraining data.

This samples random contiguous token windows from an offline token shard dataset,
decodes them back to text, and produces:
  - a JSON summary report (rates of common noise patterns)
  - a human-readable samples text file (random + flagged examples)

CPU usage: capped to 4 threads via common env vars.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
import zlib
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Cap CPU noise (no more than 4 threads in common compiled libs / tokenizers).
# Override by exporting env vars before running this script.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from ml.errors import SophiaUsageError  # noqa: E402
from ml.core.common.io import write_json_atomic  # noqa: E402
from ml.data.token_shards.shard_manifest import (  # noqa: E402
    load_manifest,
    validate_tokenizer_fingerprint,
)
from ml.data.token_shards.tokenizer_fingerprint import (  # noqa: E402
    default_modeling_tokenizer_dir,
    resolve_matching_tokenizer_dir,
)

_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HTML_TAG_RE = re.compile(
    r"(?i)<\s*/?\s*(?:"
    r"html|head|body|meta|link|title|style|script|noscript|"
    r"div|span|p|br|hr|img|a|table|tr|td|th|thead|tbody|"
    r"ul|ol|li|h1|h2|h3|h4|h5|h6|"
    r"form|input|textarea|button|select|option|iframe"
    r")\b"
)
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+);", re.IGNORECASE)

# Code: avoid obvious false positives like a natural-language line starting with "from ...".
_CODE_RE = re.compile(
    r"```|^\s*(?:"
    r"def\s+\w+|"
    r"class\s+\w+|"
    r"import\s+\w+|"
    r"from\s+\w[\w.]*\s+import\s+\w+|"
    r"#include\s+[<\"]|"
    r"package\s+\w+|"
    r"func\s+\w+"
    r")",
    flags=re.MULTILINE,
)

_REPEAT_CHAR_RE = re.compile(r"(.)\1{20,}")

# Heuristics for "web boilerplate" triage (conservative: avoid generic tokens like "目录").
_CONTACT_RE = re.compile(
    r"(?:公众号|公号|微信群|QQ群|群号|加群|进群|入群|"
    r"客服微信|联系客服|联系我们|商务合作|广告合作|投稿邮箱|"
    r"(?:微信号|VX|Vx|vx)\s*[:：]\s*[A-Za-z0-9][A-Za-z0-9_-]{3,})"
)
_NAV_RE = re.compile(
    r"(?:上一章|下一章|上一页|下一页|返回目录|章节目录|目录页|"
    r"最新章节|全文阅读|手机阅读|加入书签|加入收藏|书架|书柜|"
    r"投票|推荐票|月票|打赏|返回顶部)"
)
_WEB_FOOTER_RE = re.compile(
    r"(?:免责声明|版权声明|版权所有|版权归|转载请注明|转载来源|原文链接|"
    r"未经授权|不得转载|Copyright|All rights reserved|"
    r"ICP\s*\S{4,}|ICP备\s*\S{4,}|公安(?:网安)?备案)",
    re.IGNORECASE,
)
_DOWNLOAD_RE = re.compile(
    r"(?:点击|立即|免费)?\s*(?:下载|安装)[^\n]{0,24}(?:APP|客户端|软件|阅读器|手机版|官方)",
    re.IGNORECASE,
)

_PUNCT = set(string.punctuation) | set("，。！？；：、（）【】《》“”‘’—…·「」『』～")


def _validate_tokenizer_sha1(*, manifest, tokenizer_dir: str) -> None:
    expected = str(getattr(manifest, "tokenizer_sha1", "") or "").strip()
    if not expected:
        return
    try:
        validate_tokenizer_fingerprint(
            tokenizer_dir=str(tokenizer_dir), expected_sha1=str(expected)
        )
    except Exception as e:
        raise SophiaUsageError(str(e)) from e


def _write_json_atomic(path: str, obj: dict) -> None:
    write_json_atomic(path, obj, make_parents=True)


def safe_decode(tokenizer: Tokenizer, token_ids: np.ndarray) -> str:
    ids = [int(x) for x in token_ids.tolist()]
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        return tokenizer.decode(ids)


def _ratio(n: int, d: int) -> float:
    d = int(d)
    if d <= 0:
        return 0.0
    return float(n) / float(d)


def _cjk_ascii_counts(s: str) -> tuple[int, int]:
    cjk = 0
    ascii_alpha = 0
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            cjk += 1
        elif o < 128 and ("A" <= ch <= "Z" or "a" <= ch <= "z"):
            ascii_alpha += 1
    return cjk, ascii_alpha


def _punct_digit_space_ratios(s: str) -> tuple[float, float, float]:
    if not s:
        return 0.0, 0.0, 0.0
    punct = 0
    digits = 0
    spaces = 0
    for ch in s:
        if ch in _PUNCT:
            punct += 1
        if ch.isdigit():
            digits += 1
        if ch.isspace():
            spaces += 1
    n = len(s)
    return _ratio(punct, n), _ratio(digits, n), _ratio(spaces, n)


def _compression_ratio_utf8(s: str) -> float:
    if not s:
        return 0.0
    b = s.encode("utf-8", errors="ignore")
    if not b:
        return 0.0
    c = zlib.compress(b, level=6)
    if not c:
        return 0.0
    return float(len(b)) / float(len(c))


def _dup_4gram_ratio(token_ids: np.ndarray) -> float:
    if token_ids is None:
        return 0.0
    n = int(token_ids.size)
    if n < 8:
        return 0.0
    t = np.asarray(token_ids).astype(np.uint64, copy=False)
    # Prefer exact packing for vocab <= 2^16; fall back to a 64-bit hash combine for larger ids.
    try:
        max_id = int(t.max())
    except Exception:
        max_id = 0
    if max_id <= 0xFFFF:
        h = t[:-3] | (t[1:-2] << 16) | (t[2:-1] << 32) | (t[3:] << 48)
    else:
        a = t[:-3]
        b = t[1:-2]
        c = t[2:-1]
        d = t[3:]
        h = (
            (a * np.uint64(0x9E3779B185EBCA87))
            ^ (b * np.uint64(0xC2B2AE3D27D4EB4F))
            ^ (c * np.uint64(0x165667B19E3779F9))
            ^ (d * np.uint64(0x27D4EB2F165667C5))
        )
    if h.size <= 0:
        return 0.0
    uniq = int(np.unique(h).size)
    return 1.0 - (float(uniq) / float(h.size))


@dataclass
class Sample:
    shard: str
    start: int
    seq_len: int
    text: str
    metrics: dict[str, Any]
    reasons: list[str]


class _MemmapCache:
    def __init__(self, *, max_open: int):
        self.max_open = max(int(max_open), 1)
        self._cache: OrderedDict[int, np.memmap] = OrderedDict()

    def get(self, idx: int, path: str, dtype: np.dtype) -> np.memmap:
        idx = int(idx)
        if idx in self._cache:
            mm = self._cache.pop(idx)
            self._cache[idx] = mm
            return mm
        mm = np.memmap(path, mode="r", dtype=dtype)
        self._cache[idx] = mm
        while len(self._cache) > self.max_open:
            _, evicted_mm = self._cache.popitem(last=False)
            try:
                evicted_mm._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        return mm

    def close(self) -> None:
        for _, mm in list(self._cache.items()):
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._cache.clear()


def _analyze_text(s: str, *, token_ids: np.ndarray) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {}
    reasons: list[str] = []

    s = str(s or "")
    metrics["chars"] = int(len(s))
    metrics["newlines"] = int(s.count("\n"))

    metrics["has_url"] = bool(_URL_RE.search(s))
    metrics["has_email"] = bool(_EMAIL_RE.search(s))
    metrics["has_html"] = bool(_HTML_TAG_RE.search(s)) or bool(
        _HTML_ENTITY_RE.search(s)
    )
    metrics["has_code"] = bool(_CODE_RE.search(s))
    metrics["has_repeat_chars"] = bool(_REPEAT_CHAR_RE.search(s))

    metrics["has_contact_hint"] = bool(_CONTACT_RE.search(s))
    metrics["has_nav_hint"] = bool(_NAV_RE.search(s))
    metrics["has_footer_hint"] = bool(_WEB_FOOTER_RE.search(s))
    metrics["has_download_hint"] = bool(_DOWNLOAD_RE.search(s))

    punct_r, digit_r, space_r = _punct_digit_space_ratios(s)
    metrics["punct_ratio"] = round(punct_r, 4)
    metrics["digit_ratio"] = round(digit_r, 4)
    metrics["space_ratio"] = round(space_r, 4)

    cjk, ascii_alpha = _cjk_ascii_counts(s)
    metrics["cjk_chars"] = int(cjk)
    metrics["ascii_alpha_chars"] = int(ascii_alpha)
    metrics["cjk_ratio"] = round(_ratio(cjk, len(s)), 4)

    # Mojibake: count UTF-8 replacement chars only (do NOT treat '?' as bad).
    bad = s.count("\ufffd")
    metrics["bad_char_count"] = int(bad)
    metrics["bad_char_ratio"] = round(_ratio(bad, len(s)), 6)

    cr = _compression_ratio_utf8(s)
    metrics["compression_ratio"] = round(cr, 4)

    if token_ids is not None and token_ids.size > 0:
        uniq = int(np.unique(token_ids).size)
        metrics["uniq_token_ratio"] = round(float(uniq) / float(token_ids.size), 4)
        metrics["dup_4gram_ratio"] = round(_dup_4gram_ratio(token_ids), 4)

    # Reasons (heuristics for visual triage).
    if metrics["has_url"]:
        reasons.append("url")
    if metrics["has_email"]:
        reasons.append("email")
    if metrics["has_html"]:
        reasons.append("html")
    if metrics["has_code"]:
        reasons.append("code")
    if metrics["has_contact_hint"]:
        reasons.append("contact")
    if metrics["has_nav_hint"]:
        reasons.append("nav")
    if metrics["has_footer_hint"]:
        reasons.append("footer")
    if metrics["has_download_hint"]:
        reasons.append("download")

    if (
        float(metrics.get("bad_char_ratio", 0.0)) >= 0.002
        or int(metrics.get("bad_char_count", 0)) >= 5
    ):
        reasons.append("mojibake")
    if float(metrics.get("punct_ratio", 0.0)) >= 0.35:
        reasons.append("punct_high")
    if float(metrics.get("digit_ratio", 0.0)) >= 0.35:
        reasons.append("digit_high")
    if float(metrics.get("compression_ratio", 0.0)) >= 6.0:
        reasons.append("repetitive")
    if float(metrics.get("uniq_token_ratio", 1.0)) <= 0.10:
        reasons.append("low_token_diversity")
    if float(metrics.get("dup_4gram_ratio", 0.0)) >= 0.25:
        reasons.append("ngram_dup_high")
    return metrics, reasons


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visual audit for token-shard pretraining data."
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=os.path.join("dataset", "pretrain_tokens", "train", "manifest.json"),
    )
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help=(
            "Optional tokenizer directory (expects tokenizer.json). "
            "If omitted, prefer a dataset-local tokenizer bundle and otherwise use the bundled runtime tokenizer."
        ),
    )
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cache_shards", type=int, default=16, help="Max open shard memmaps."
    )
    p.add_argument(
        "--random_examples",
        type=int,
        default=25,
        help="How many random examples to include.",
    )
    p.add_argument(
        "--max_chars_per_example",
        type=int,
        default=2000,
        help="Truncate decoded text for samples output.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("out", "reports"),
        help="Directory to write report + samples.",
    )
    args = p.parse_args()

    manifest_path = os.path.abspath(str(args.manifest))
    if not os.path.exists(manifest_path):
        raise SophiaUsageError(f"Missing manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    manifest_dir = os.path.dirname(manifest_path) or "."

    tok_path = resolve_matching_tokenizer_dir(
        manifest_path=str(manifest_path),
        expected_sha1=str(manifest.tokenizer_sha1),
        explicit_tokenizer_dir=str(args.tokenizer_path),
        default_tokenizer_dir=default_modeling_tokenizer_dir(),
    )
    _validate_tokenizer_sha1(manifest=manifest, tokenizer_dir=tok_path)
    tokenizer = Tokenizer.from_file(str(Path(tok_path) / "tokenizer.json"))

    seq_len = max(int(args.seq_len), 16)
    target_samples = max(int(args.samples), 1)
    rng = np.random.default_rng(int(args.seed))

    dtype_str = str(manifest.dtype)
    if dtype_str == "uint16":
        dtype = np.dtype("<u2")
    elif dtype_str == "uint32":
        dtype = np.dtype("<u4")
    elif dtype_str == "int32":
        dtype = np.dtype("<i4")
    elif dtype_str == "int64":
        dtype = np.dtype("<i8")
    else:
        raise SophiaUsageError(f"[ERR] Unsupported manifest dtype: {dtype_str!r}")
    shard_paths: list[str] = [
        os.path.join(manifest_dir, s.path) for s in manifest.shards
    ]
    shard_tokens = np.asarray([int(s.tokens) for s in manifest.shards], dtype=np.int64)

    # Sample windows uniformly across all tokens (proportional to #windows per shard).
    windows = np.maximum(shard_tokens - seq_len, 0)
    if int(windows.sum()) <= 0:
        raise SophiaUsageError(
            f"No valid windows for seq_len={seq_len} (check manifest/shards)."
        )
    probs = windows.astype(np.float64)
    probs /= float(probs.sum())

    cache = _MemmapCache(max_open=int(args.cache_shards))

    t0 = time.time()
    stats = Counter()
    metric_sums = Counter()
    metric_max: dict[str, float] = {}
    flagged: dict[str, list[Sample]] = {
        k: []
        for k in (
            "url",
            "html",
            "contact",
            "nav",
            "footer",
            "download",
            "code",
            "mojibake",
            "repetitive",
        )
    }
    random_pool: list[Sample] = []

    def consider_flag(sample: Sample) -> None:
        for k in list(flagged.keys()):
            if k in sample.reasons:
                flagged[k].append(sample)

    def update_metric_max(metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                fv = float(v)
                if k not in metric_max or fv > float(metric_max[k]):
                    metric_max[k] = fv

    ok = 0
    attempts = 0
    max_attempts = target_samples * 4
    while ok < target_samples and attempts < max_attempts:
        attempts += 1
        shard_idx = int(rng.choice(len(shard_paths), p=probs))
        path = shard_paths[shard_idx]
        mm = cache.get(shard_idx, path, dtype)
        n = int(mm.shape[0])
        max_start = n - seq_len
        if max_start <= 0:
            continue
        start = int(rng.integers(0, max_start + 1))
        toks = np.asarray(mm[start : start + seq_len]).astype(np.int32, copy=True)
        text = safe_decode(tokenizer, toks)
        metrics, reasons = _analyze_text(text, token_ids=toks)
        sample = Sample(
            shard=str(os.path.basename(path)),
            start=int(start),
            seq_len=int(seq_len),
            text=str(text),
            metrics=metrics,
            reasons=reasons,
        )
        ok += 1

        stats["samples"] += 1
        for r in reasons:
            stats[f"reason_{r}"] += 1
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_sums[k] += float(v)
        update_metric_max(metrics)

        if len(random_pool) < int(args.random_examples):
            random_pool.append(sample)
        consider_flag(sample)

        if ok % 200 == 0 or ok == target_samples:
            elapsed = time.time() - t0
            rate = ok / max(elapsed, 1e-6)
            print(
                f"[AUDIT] samples={ok}/{target_samples} rate={rate:.1f}/s elapsed={elapsed:.1f}s",
                flush=True,
            )

    cache.close()

    if ok <= 0:
        raise SophiaUsageError("No samples collected.")

    def top_by(metric: str, items: list[Sample], n: int) -> list[Sample]:
        n = max(int(n), 0)
        if n <= 0:
            return []

        def key_fn(s: Sample) -> float:
            v = s.metrics.get(metric)
            return float(v) if isinstance(v, (int, float)) else 0.0

        return sorted(items, key=key_fn, reverse=True)[:n]

    # Keep the samples output compact but useful.
    top_n = 10
    curated: dict[str, list[Sample]] = {
        "random": random_pool[: int(args.random_examples)],
        "url": top_by("chars", flagged["url"], top_n),
        "html": top_by("chars", flagged["html"], top_n),
        "contact": top_by("chars", flagged["contact"], top_n),
        "nav": top_by("chars", flagged["nav"], top_n),
        "footer": top_by("chars", flagged["footer"], top_n),
        "download": top_by("chars", flagged["download"], top_n),
        "code": top_by("chars", flagged["code"], top_n),
        "mojibake": top_by("bad_char_ratio", flagged["mojibake"], top_n),
        "repetitive": top_by("compression_ratio", flagged["repetitive"], top_n),
    }

    avg_metrics: dict[str, float] = {}
    for k, total in metric_sums.items():
        avg_metrics[k] = round(float(total) / float(ok), 6)

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_report = os.path.join(out_dir, f"token_shards_audit_{run_id}.json")
    out_samples = os.path.join(out_dir, f"token_shards_audit_{run_id}.samples.txt")
    out_latest = os.path.join(out_dir, "token_shards_audit.latest.json")
    out_latest_samples = os.path.join(out_dir, "token_shards_audit.latest.samples.txt")

    meta = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path.replace("\\", "/"),
        "tokenizer_path": tok_path.replace("\\", "/"),
        "seq_len": int(seq_len),
        "samples": int(ok),
        "seed": int(args.seed),
        "thresholds": {
            "punct_ratio_ge": 0.35,
            "digit_ratio_ge": 0.35,
            "bad_char_ratio_ge": 0.002,
            "bad_char_count_ge": 5,
            "compression_ratio_ge": 6.0,
            "uniq_token_ratio_le": 0.10,
            "dup_4gram_ratio_ge": 0.25,
        },
        "stats": dict(stats),
        "avg_metrics": avg_metrics,
        "max_metrics": {k: round(float(v), 6) for k, v in metric_max.items()},
        "curated_counts": {k: int(len(v)) for k, v in curated.items()},
    }

    _write_json_atomic(out_report, meta)
    _write_json_atomic(out_latest, meta)

    def fmt_sample(s: Sample) -> str:
        txt = s.text
        max_chars = int(args.max_chars_per_example)
        if max_chars > 0 and len(txt) > max_chars:
            txt = txt[:max_chars] + "\n...[truncated]..."
        return (
            f"shard={s.shard} start={s.start} seq_len={s.seq_len}\n"
            f"reasons={','.join(s.reasons) if s.reasons else '-'}\n"
            f"metrics={json.dumps(s.metrics, ensure_ascii=False)}\n"
            f"--- text ---\n{txt}\n"
        )

    with open(out_samples, "w", encoding="utf-8") as f:
        f.write("=== Token Shards Visual Audit ===\n")
        f.write(f"time_utc: {meta['time_utc']}\n")
        f.write(f"manifest: {meta['manifest']}\n")
        f.write(f"tokenizer: {meta['tokenizer_path']}\n")
        f.write(f"seq_len: {seq_len}\n")
        f.write(f"samples: {ok}\n\n")
        for section, items in curated.items():
            f.write(f"\n===== [{section.upper()}] ({len(items)}) =====\n\n")
            for i, s in enumerate(items, 1):
                f.write(f"### {section} #{i}\n")
                f.write(fmt_sample(s))

    Path(out_latest_samples).unlink(missing_ok=True)
    os.replace(out_samples, out_latest_samples)

    print(f"[OK] report: {out_report}")
    print(f"[OK] samples: {out_latest_samples}")


if __name__ == "__main__":
    main()
