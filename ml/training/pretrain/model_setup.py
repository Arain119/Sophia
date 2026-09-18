from __future__ import annotations

import torch

from ml.modeling import SophiaDecoderConfig, SophiaDecoder
from ml.integrations.adapters.hf.tokenizer import (
    load_local_tokenizer,
    tokenizer_vocab_size,
)
from ml.training.pretrain.resources import PretrainTokenizerLike
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.profiles import resolve_pretrain_profile
from ml.training.pretrain.model_runtime_controls import (
    require_runtime_recipe_control,
)


def _positive_int_or_default(value: object, *, default: int) -> int:
    if value is None or value == "":
        return int(default)
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"expected integer configuration value, got {value!r}") from exc
    if resolved < 0:
        raise ValueError(f"configuration value must be >= 0, got {resolved}")
    return int(default) if resolved <= 0 else int(resolved)


def load_tokenizer(
    tok_path: str,
    *,
    model_max_length: int | None = None,
) -> PretrainTokenizerLike:
    return load_local_tokenizer(
        tok_path,
        padding_side="right",
        truncation_side="left",
        model_max_length=model_max_length,
    )


def build_decoder_config(
    *,
    args: PretrainRunConfig,
    tokenizer: PretrainTokenizerLike,
) -> tuple[SophiaDecoderConfig, int]:
    profile = resolve_pretrain_profile(args)

    tok_vocab = _positive_int_or_default(
        tokenizer_vocab_size(tokenizer),
        default=0,
    )
    if tok_vocab <= 0:
        raise ValueError("Tokenizer vocab_size must be a positive integer.")
    if tok_vocab != int(profile.model.vocab_size):
        raise ValueError(
            "Tokenizer vocab_size mismatch.\n"
            f"tokenizer_vocab_size={tok_vocab}\n"
            f"expected_vocab_size={int(profile.model.vocab_size)}\n"
            "This usually means your tokenizer files are stale/wrong.\n"
            "Suggestion: point --tokenizer_path to the matching tokenizer, or regenerate the HF tokenizer."
        )

    vocab_size = int(tok_vocab)

    special_token_kwargs: dict[str, int] = {}
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if not isinstance(pad_token_id, int) and isinstance(eos_token_id, int):
        pad_token_id = int(eos_token_id)
    for key in ("bos_token_id", "eos_token_id", "unk_token_id"):
        val = getattr(tokenizer, key)
        if isinstance(val, int):
            special_token_kwargs[key] = int(val)
    if isinstance(pad_token_id, int):
        special_token_kwargs["pad_token_id"] = int(pad_token_id)

    max_seq_len = _positive_int_or_default(
        args.max_seq_len,
        default=int(profile.training.max_seq_len),
    )
    max_batch_size = _positive_int_or_default(args.batch_size, default=4)
    cfg = profile.model.with_overrides(
        vocab_size=int(vocab_size),
        max_seq_len=int(max_seq_len),
        max_batch_size=int(max_batch_size),
    ).to_config()
    cfg = SophiaDecoderConfig.from_object(cfg)
    for key, value in special_token_kwargs.items():
        setattr(cfg, key, int(value))
    return cfg, int(vocab_size)


def sync_runtime_batch_capacity(*, model: torch.nn.Module, args: PretrainRunConfig) -> None:
    max_batch_size = _positive_int_or_default(
        args.batch_size,
        default=_positive_int_or_default(
            getattr(getattr(model, "config", None), "max_batch_size", 4),
            default=4,
        ),
    )
    require_runtime_recipe_control(model).sync_runtime_batch_capacity(
        int(max_batch_size)
    )


def initialize_model(
    *,
    args: PretrainRunConfig,
    decoder_config: SophiaDecoderConfig,
    device: torch.device,
    base_dtype: torch.dtype,
) -> torch.nn.Module:
    configured_max_seq_len = _positive_int_or_default(
        getattr(decoder_config, "max_seq_len", 0),
        default=0,
    )
    requested_runtime_max_seq_len = _positive_int_or_default(
        args.seq_len,
        default=int(configured_max_seq_len),
    )
    runtime_max_seq_len = int(
        min(requested_runtime_max_seq_len, configured_max_seq_len)
    )
    decoder_config.return_logits_in_train = False
    decoder_config.use_cache = False
    decoder_config.max_batch_size = int(decoder_config.max_batch_size)
    return SophiaDecoder(
        decoder_config,
        runtime_max_seq_len=int(runtime_max_seq_len),
    ).to(device=device, dtype=base_dtype)


__all__ = [
    "build_decoder_config",
    "initialize_model",
    "load_tokenizer",
    "sync_runtime_batch_capacity",
]
