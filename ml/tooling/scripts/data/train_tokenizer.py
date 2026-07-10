from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from math import floor
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import Sequence as NormalizerSequence
from transformers import PreTrainedTokenizerFast

from ml.core.common.io import write_json_atomic
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1


CORE_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|unk|>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜latest_reminder｜>",
    "<think>",
    "</think>",
)

ADDITIONAL_SPECIAL_TOKENS = CORE_SPECIAL_TOKENS[4:]

EVAL_SAMPLES = {
    "zh_short": "你好，世界。这个 tokenizer 应该稳定处理中英文混排。",
    "en_short": "A strong tokenizer should keep common English words compact.",
    "math_latex": "设 f(x)=\\frac{x^2+1}{2}，求 f'(x) 并解释步骤。",
    "code_python": "from collections import Counter\nprint(Counter(['token', 'token']))\n",
    "dialog": "<｜User｜>请简要解释梯度裁剪。<｜Assistant｜>梯度裁剪用于限制更新幅度。",
    "thinking": "<think>先列出约束，再给出结论。</think>",
    "punctuation": "URLs, e-mail, JSON: {\"ok\": true, \"n\": 12345}.",
    "rare_unicode": "emoji🙂、全角ＡＢＣ、日文かなカナ、accent café naïve.",
}

DEFAULT_PRETRAIN_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_SFT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TEXT_CHARS = 8192
DEFAULT_BUCKET_MIN_BYTES = 1 * 1024 * 1024


@dataclass
class TrainingTextStats:
    pretrain_texts: int = 0
    pretrain_bytes: int = 0
    sft_texts: int = 0
    sft_bytes: int = 0
    truncated_texts: int = 0
    buckets: dict[str, dict[str, int]] | None = None

    @property
    def total_texts(self) -> int:
        return int(self.pretrain_texts + self.sft_texts)

    @property
    def total_bytes(self) -> int:
        return int(self.pretrain_bytes + self.sft_bytes)

    def as_dict(self) -> dict[str, object]:
        return {
            "pretrain_texts": int(self.pretrain_texts),
            "pretrain_bytes": int(self.pretrain_bytes),
            "sft_texts": int(self.sft_texts),
            "sft_bytes": int(self.sft_bytes),
            "total_texts": int(self.total_texts),
            "total_bytes": int(self.total_bytes),
            "truncated_texts": int(self.truncated_texts),
            "buckets": dict(self.buckets or {}),
        }

    def add(self, *, source: str, bucket: str, text_bytes: int) -> None:
        if source == "pretrain":
            self.pretrain_texts += 1
            self.pretrain_bytes += int(text_bytes)
        elif source == "sft":
            self.sft_texts += 1
            self.sft_bytes += int(text_bytes)
        else:
            raise ValueError(f"unknown source: {source}")
        if self.buckets is None:
            self.buckets = {}
        item = self.buckets.setdefault(str(bucket), {"texts": 0, "bytes": 0})
        item["texts"] += 1
        item["bytes"] += int(text_bytes)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _parquet_text_column(path: Path) -> str:
    parquet_file = pq.ParquetFile(path)
    names = set(parquet_file.schema.names)
    return "text" if "text" in names else ("content" if "content" in names else "")


def _iter_parquet_file_texts(path: Path) -> Iterator[str]:
    text_column = _parquet_text_column(path)
    if not text_column:
        return
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=[text_column], batch_size=8192):
        column = batch.column(0)
        for value in column.to_pylist():
            text = str(value or "").strip()
            if text:
                yield text


def _iter_parquet_texts(root: Path) -> Iterator[str]:
    for path in sorted(root.rglob("*.parquet")):
        yield from _iter_parquet_file_texts(path)


def _iter_sft_texts(root: Path) -> Iterator[str]:
    for path in sorted(root.rglob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                conversations = payload.get("conversations")
                if not isinstance(conversations, list):
                    conversations = payload.get("messages")
                if isinstance(conversations, list):
                    messages = conversations
                else:
                    messages = [payload]
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    content = str(
                        message.get("content")
                        or message.get("text")
                        or message.get("value")
                        or ""
                    ).strip()
                    if content:
                        yield content


def _iter_sft_file_texts(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            conversations = payload.get("conversations")
            if not isinstance(conversations, list):
                conversations = payload.get("messages")
            messages = conversations if isinstance(conversations, list) else [payload]
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = str(
                    message.get("content")
                    or message.get("text")
                    or message.get("value")
                    or ""
                ).strip()
                if content:
                    yield content


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _trim_to_utf8_budget(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return text
    if _utf8_len(text) <= max_bytes:
        return text
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore").strip()


def _bounded_texts(
    iterator: Iterator[str],
    *,
    source: str,
    bucket: str = "",
    stats: TrainingTextStats,
    max_texts: int,
    max_bytes: int,
    max_text_chars: int,
) -> Iterator[str]:
    source_texts = 0
    source_bytes = 0
    for text in iterator:
        if max_texts > 0 and source_texts >= max_texts:
            return
        if max_text_chars > 0 and len(text) > max_text_chars:
            text = text[:max_text_chars].strip()
            stats.truncated_texts += 1
        text_bytes = _utf8_len(text)
        if max_bytes > 0 and source_bytes + text_bytes > max_bytes:
            text = _trim_to_utf8_budget(text, max_bytes - source_bytes)
            stats.truncated_texts += 1
            if not text:
                return
            text_bytes = _utf8_len(text)
        if text_bytes <= 0:
            continue
        yield text
        source_texts += 1
        source_bytes += text_bytes
        stats.add(source=source, bucket=bucket or source, text_bytes=text_bytes)
        if max_bytes > 0 and source_bytes >= max_bytes:
            return


def _bucket_for_pretrain_file(path: Path, pretrain_dir: Path) -> str:
    rel = path.relative_to(pretrain_dir)
    parts = rel.parts
    if len(parts) <= 2:
        return "pretrain/" + (parts[0] if parts else "root")
    return "pretrain/" + "/".join(parts[1:-1])


def _pretrain_bucket_files(pretrain_dir: Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = {}
    for path in sorted(pretrain_dir.rglob("*.parquet")):
        try:
            if not _parquet_text_column(path):
                continue
        except Exception:
            continue
        bucket = _bucket_for_pretrain_file(path, pretrain_dir)
        buckets.setdefault(bucket, []).append(path)
    return buckets


def _sft_bucket_files(sft_dir: Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = {}
    for path in sorted(sft_dir.rglob("*.jsonl")):
        stem = path.stem.lower()
        if stem in {"train", "val", "test"}:
            bucket = f"sft/{stem}"
        else:
            bucket = "sft/" + path.relative_to(sft_dir).with_suffix("").as_posix()
        buckets.setdefault(bucket, []).append(path)
    return buckets


def _allocate_bucket_budgets(
    total_bytes: int,
    bucket_sizes: dict[str, int],
    *,
    min_bucket_bytes: int,
) -> dict[str, int]:
    if total_bytes <= 0:
        return {bucket: 0 for bucket in bucket_sizes}
    buckets = sorted(bucket_sizes)
    if not buckets:
        return {}
    floor_budget = min(int(min_bucket_bytes), max(0, int(total_bytes) // len(buckets)))
    budgets = {bucket: floor_budget for bucket in buckets}
    remaining = int(total_bytes) - sum(budgets.values())
    if remaining <= 0:
        return budgets
    total_size = sum(max(0, int(bucket_sizes[bucket])) for bucket in buckets)
    if total_size <= 0:
        base = remaining / float(len(buckets))
        for bucket in buckets:
            budgets[bucket] += int(floor(base))
        remainder = int(total_bytes) - sum(budgets.values())
        for bucket in buckets[:remainder]:
            budgets[bucket] += 1
        return budgets
    fractional: list[tuple[float, str]] = []
    for bucket in buckets:
        exact = remaining * (max(0, int(bucket_sizes[bucket])) / float(total_size))
        add = int(floor(exact))
        budgets[bucket] += add
        fractional.append((exact - add, bucket))
    remainder = int(total_bytes) - sum(budgets.values())
    for _frac, bucket in sorted(fractional, reverse=True)[:remainder]:
        budgets[bucket] += 1
    return budgets


def _bucket_file_sizes(buckets: dict[str, list[Path]]) -> dict[str, int]:
    return {
        bucket: sum(path.stat().st_size for path in files if path.exists())
        for bucket, files in buckets.items()
    }


def _iter_round_robin_texts(
    files: list[Path],
    *,
    kind: str,
) -> Iterator[str]:
    iterators = []
    for path in files:
        iterator = _iter_parquet_file_texts(path) if kind == "pretrain" else _iter_sft_file_texts(path)
        iterators.append(iter(iterator))
    while iterators:
        next_iterators = []
        for iterator in iterators:
            try:
                yield next(iterator)
                next_iterators.append(iterator)
            except StopIteration:
                continue
        iterators = next_iterators


def iter_stratified_training_texts(
    *,
    pretrain_dir: Path,
    sft_dir: Path,
    pretrain_max_texts: int,
    sft_max_texts: int,
    pretrain_max_bytes: int,
    sft_max_bytes: int,
    max_text_chars: int,
    bucket_min_bytes: int,
    stats: TrainingTextStats,
) -> Iterator[str]:
    pretrain_buckets = _pretrain_bucket_files(pretrain_dir)
    pretrain_budgets = _allocate_bucket_budgets(
        pretrain_max_bytes,
        _bucket_file_sizes(pretrain_buckets),
        min_bucket_bytes=bucket_min_bytes,
    )
    for bucket in sorted(pretrain_buckets):
        yield from _bounded_texts(
            _iter_round_robin_texts(pretrain_buckets[bucket], kind="pretrain"),
            source="pretrain",
            bucket=bucket,
            stats=stats,
            max_texts=pretrain_max_texts,
            max_bytes=pretrain_budgets[bucket],
            max_text_chars=max_text_chars,
        )

    sft_buckets = _sft_bucket_files(sft_dir)
    sft_budgets = _allocate_bucket_budgets(
        sft_max_bytes,
        _bucket_file_sizes(sft_buckets),
        min_bucket_bytes=bucket_min_bytes,
    )
    for bucket in sorted(sft_buckets):
        yield from _bounded_texts(
            _iter_round_robin_texts(sft_buckets[bucket], kind="sft"),
            source="sft",
            bucket=bucket,
            stats=stats,
            max_texts=sft_max_texts,
            max_bytes=sft_budgets[bucket],
            max_text_chars=max_text_chars,
        )


def iter_training_texts(
    *,
    pretrain_dir: Path,
    sft_dir: Path,
    pretrain_max_texts: int,
    sft_max_texts: int,
    pretrain_max_bytes: int,
    sft_max_bytes: int,
    max_text_chars: int,
    stats: TrainingTextStats,
    bucket_min_bytes: int = 0,
) -> Iterator[str]:
    yield from _bounded_texts(
        _iter_parquet_texts(pretrain_dir),
        source="pretrain",
        stats=stats,
        max_texts=pretrain_max_texts,
        max_bytes=pretrain_max_bytes,
        max_text_chars=max_text_chars,
    )
    yield from _bounded_texts(
        _iter_sft_texts(sft_dir),
        source="sft",
        stats=stats,
        max_texts=sft_max_texts,
        max_bytes=sft_max_bytes,
        max_text_chars=max_text_chars,
    )


def build_tokenizer(*, vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>", byte_fallback=True))
    tokenizer.normalizer = NormalizerSequence([])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(r"\p{N}{1,3}", behavior="isolated"),
            pre_tokenizers.Split(
                r"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
                behavior="isolated",
            ),
            pre_tokenizers.Split(
                r"[!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )
    tokenizer.post_processor = processors.ByteLevel(
        add_prefix_space=True,
        trim_offsets=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel(
        add_prefix_space=True,
        trim_offsets=True,
        use_regex=True,
    )
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=list(CORE_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        max_token_length=64,
    )
    tokenizer._trainer = trainer  # type: ignore[attr-defined]
    return tokenizer


def _encoding_stats(*, fast: PreTrainedTokenizerFast, sample: str) -> dict[str, object]:
    ids = fast.encode(str(sample), add_special_tokens=False)
    tokens = fast.convert_ids_to_tokens(ids)
    decoded = fast.decode(ids, skip_special_tokens=False)
    chars = len(str(sample))
    token_count = len(ids)
    return {
        "chars": int(chars),
        "tokens": int(token_count),
        "chars_per_token": (
            0.0 if token_count <= 0 else round(float(chars) / float(token_count), 4)
        ),
        "ids": ids[:96],
        "tokens_preview": [str(token) for token in tokens[:32]],
        "roundtrip_ok": bool(decoded == sample),
        "decoded": decoded,
    }


def _special_token_payload(fast: PreTrainedTokenizerFast) -> dict[str, object]:
    return {
        "all_special_tokens": list(fast.all_special_tokens),
        "all_special_ids": [int(x) for x in fast.all_special_ids],
        "core_special_token_ids": {
            str(token): int(fast.convert_tokens_to_ids(token))
            for token in CORE_SPECIAL_TOKENS
        },
    }


def write_tokenizer_bundle(
    *,
    tokenizer: Tokenizer,
    output_dir: Path,
    chat_template_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "additional_special_tokens": list(ADDITIONAL_SPECIAL_TOKENS),
        "added_tokens_decoder": {},
        "bos_token": "<｜begin▁of▁sentence｜>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<｜end▁of▁sentence｜>",
        "model_max_length": 4096,
        "pad_token": "<|pad|>",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|unk|>",
        "chat_template": chat_template,
    }
    special_tokens_map = {
        "additional_special_tokens": list(ADDITIONAL_SPECIAL_TOKENS),
        "bos_token": "<｜begin▁of▁sentence｜>",
        "eos_token": "<｜end▁of▁sentence｜>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
    }
    write_json_atomic(str(output_dir / "tokenizer_config.json"), tokenizer_config)
    write_json_atomic(str(output_dir / "special_tokens_map.json"), special_tokens_map)
    (output_dir / "chat_template.jinja").write_text(chat_template, encoding="utf-8")

    fast = PreTrainedTokenizerFast.from_pretrained(str(output_dir), local_files_only=True)
    fast.save_pretrained(str(output_dir))


def _validate_bundle(
    output_dir: Path,
    *,
    expected_vocab_size: int,
    training_stats: TrainingTextStats,
) -> dict[str, object]:
    fast = PreTrainedTokenizerFast.from_pretrained(str(output_dir), local_files_only=True)
    vocab = fast.get_vocab()
    vocab_size = int(len(vocab))
    if vocab_size != int(expected_vocab_size):
        raise RuntimeError(
            f"unexpected vocab_size: expected={int(expected_vocab_size)} got={vocab_size}"
        )
    missing = [token for token in CORE_SPECIAL_TOKENS if token not in vocab]
    if missing:
        raise RuntimeError(f"missing special tokens: {missing}")
    unexpected_special = [
        token for token in fast.all_special_tokens if token not in CORE_SPECIAL_TOKENS
    ]
    if unexpected_special:
        raise RuntimeError(f"unexpected special tokens: {unexpected_special}")
    evals = {
        name: _encoding_stats(fast=fast, sample=sample)
        for name, sample in EVAL_SAMPLES.items()
    }
    fertility = [
        float(item["chars_per_token"])
        for item in evals.values()
        if isinstance(item.get("chars_per_token"), (int, float))
    ]
    roundtrip_failures = [
        name for name, item in evals.items() if not bool(item.get("roundtrip_ok"))
    ]
    token_id_counts = Counter()
    for item in evals.values():
        for token_id in item.get("ids", []):
            token_id_counts[int(token_id)] += 1
    return {
        "vocab_size": vocab_size,
        "training_input": training_stats.as_dict(),
        "protocol": {
            "core_special_tokens": list(CORE_SPECIAL_TOKENS),
            "additional_special_tokens": list(ADDITIONAL_SPECIAL_TOKENS),
            "reasoning_tokens_enabled": True,
        },
        "special_tokens": _special_token_payload(fast),
        "evaluation": {
            "samples": evals,
            "roundtrip_failures": roundtrip_failures,
            "mean_chars_per_token": (
                0.0
                if not fertility
                else round(sum(fertility) / float(len(fertility)), 4)
            ),
            "min_chars_per_token": 0.0 if not fertility else round(min(fertility), 4),
            "max_chars_per_token": 0.0 if not fertility else round(max(fertility), 4),
            "top_eval_token_ids": [
                {"id": int(token_id), "count": int(count)}
                for token_id, count in token_id_counts.most_common(32)
            ],
        },
        "tokenizer_bundle_sha1": compute_tokenizer_bundle_sha1(str(output_dir)),
    }


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Train the Sophia byte-level BPE tokenizer.")
    parser.add_argument("--pretrain_dir", type=str, default=str(root / "dataset" / "pretrain"))
    parser.add_argument("--sft_dir", type=str, default=str(root / "dataset" / "sft"))
    parser.add_argument("--output_dir", type=str, default=str(root / "out" / "tokenizer_sophia_no_tools"))
    parser.add_argument("--vocab_size", type=int, default=49152)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="stratified",
        choices=["stratified", "sequential"],
        help="Use stratified bucket sampling by default; sequential preserves legacy ordering.",
    )
    parser.add_argument(
        "--pretrain_max_texts",
        type=int,
        default=0,
        help="0 means no text-count cap for pretrain data. In stratified mode this is per bucket.",
    )
    parser.add_argument(
        "--sft_max_texts",
        type=int,
        default=0,
        help="0 means no text-count cap for SFT data. In stratified mode this is per bucket.",
    )
    parser.add_argument(
        "--max_texts",
        type=int,
        default=0,
        help="Backward-compatible alias applied to pretrain and SFT text-count caps.",
    )
    parser.add_argument(
        "--pretrain_max_bytes",
        type=int,
        default=DEFAULT_PRETRAIN_MAX_BYTES,
        help="UTF-8 bytes to sample from pretrain data. Use 0 only with --allow_unbounded.",
    )
    parser.add_argument(
        "--sft_max_bytes",
        type=int,
        default=DEFAULT_SFT_MAX_BYTES,
        help="UTF-8 bytes to sample from SFT data. Use 0 only with --allow_unbounded.",
    )
    parser.add_argument(
        "--max_text_chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help="Truncate each training text to this many characters. Use 0 only with --allow_unbounded.",
    )
    parser.add_argument(
        "--bucket_min_bytes",
        type=int,
        default=DEFAULT_BUCKET_MIN_BYTES,
        help="Minimum coverage budget per stratified bucket before proportional allocation.",
    )
    parser.add_argument(
        "--allow_unbounded",
        action="store_true",
        help="Allow byte/text-char caps to be set to 0 for large-memory offline training.",
    )
    return parser


def _resolve_limit_args(args: argparse.Namespace) -> dict[str, int]:
    pretrain_max_texts = int(args.pretrain_max_texts)
    sft_max_texts = int(args.sft_max_texts)
    if int(args.max_texts) > 0:
        pretrain_max_texts = int(args.max_texts)
        sft_max_texts = int(args.max_texts)
    limits = {
        "pretrain_max_texts": pretrain_max_texts,
        "sft_max_texts": sft_max_texts,
        "pretrain_max_bytes": int(args.pretrain_max_bytes),
        "sft_max_bytes": int(args.sft_max_bytes),
        "max_text_chars": int(args.max_text_chars),
        "bucket_min_bytes": int(args.bucket_min_bytes),
    }
    if not bool(args.allow_unbounded):
        unbounded = [
            name
            for name in ("pretrain_max_bytes", "sft_max_bytes", "max_text_chars")
            if int(limits[name]) <= 0
        ]
        if unbounded:
            raise SystemExit(
                "refusing unbounded tokenizer training without --allow_unbounded: "
                + ", ".join(unbounded)
            )
    return limits


def main() -> None:
    args = build_parser().parse_args()
    root = _repo_root()
    output_dir = Path(args.output_dir).resolve()
    limits = _resolve_limit_args(args)
    training_stats = TrainingTextStats()
    tokenizer = build_tokenizer(
        vocab_size=int(args.vocab_size),
        min_frequency=int(args.min_frequency),
    )
    trainer = tokenizer._trainer  # type: ignore[attr-defined]
    text_iterator = (
        iter_stratified_training_texts
        if str(args.sampling_strategy) == "stratified"
        else iter_training_texts
    )
    tokenizer.train_from_iterator(
        text_iterator(
            pretrain_dir=Path(args.pretrain_dir).resolve(),
            sft_dir=Path(args.sft_dir).resolve(),
            pretrain_max_texts=limits["pretrain_max_texts"],
            sft_max_texts=limits["sft_max_texts"],
            pretrain_max_bytes=limits["pretrain_max_bytes"],
            sft_max_bytes=limits["sft_max_bytes"],
            max_text_chars=limits["max_text_chars"],
            bucket_min_bytes=limits["bucket_min_bytes"],
            stats=training_stats,
        ),
        trainer=trainer,
        length=None,
    )
    if training_stats.total_texts <= 0:
        raise RuntimeError("no tokenizer training texts were read")
    write_tokenizer_bundle(
        tokenizer=tokenizer,
        output_dir=output_dir,
        chat_template_path=root / "ml" / "modeling" / "text" / "chat_template.jinja",
    )
    report = _validate_bundle(
        output_dir,
        expected_vocab_size=int(args.vocab_size),
        training_stats=training_stats,
    )
    write_json_atomic(str(output_dir / "tokenizer_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
