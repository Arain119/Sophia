from __future__ import annotations

from typing import Protocol

from ml.runtime.inference.eval.common import torch
from ml.modeling.input_mask import is_all_ones_mask


class PaddingTokenizerLike(Protocol):
    pad_token_id: int | None


def is_padding_mask(
    attention_mask: torch.Tensor,
) -> tuple[bool, bool, torch.Tensor | None]:
    if attention_mask.dim() != 2:
        return False, False, None

    mask = attention_mask if attention_mask.dtype == torch.bool else (attention_mask != 0)
    if mask.numel() == 0:
        return True, True, torch.zeros(
            (int(mask.size(0)),),
            dtype=torch.int32,
            device=mask.device,
        )

    seq_len = int(mask.size(1))
    lens = mask.to(dtype=torch.int32).sum(dim=1)
    if int(lens.min().item()) <= 0:
        return False, False, None

    ar = torch.arange(seq_len, device=mask.device, dtype=torch.int32).view(1, -1)
    exp_left = ar >= (seq_len - lens).view(-1, 1)
    exp_right = ar < lens.view(-1, 1)
    left_ok = (mask == exp_left).all(dim=1)
    right_ok = (mask == exp_right).all(dim=1)
    ok = left_ok | right_ok
    if not bool(ok.all().item()):
        return False, False, None

    is_left = bool(left_ok.all().item())
    return True, is_left, lens


def right_pad_to_left_pad(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = attention_mask if attention_mask.dtype == torch.bool else (attention_mask != 0)
    lens = mask.to(dtype=torch.int32).sum(dim=1)
    bsz, seq_len = int(input_ids.size(0)), int(input_ids.size(1))

    out_ids = torch.full_like(input_ids, int(pad_token_id))
    out_mask = torch.zeros_like(mask)
    for b in range(bsz):
        valid_len = int(lens[b].item())
        if valid_len <= 0:
            continue
        if bool(mask[b, -1].item()):
            out_ids[b] = input_ids[b]
            out_mask[b] = mask[b]
            continue
        out_ids[b, seq_len - valid_len : seq_len] = input_ids[b, :valid_len]
        out_mask[b, seq_len - valid_len : seq_len] = True

    out_attn = (
        out_mask
        if attention_mask.dtype == torch.bool
        else out_mask.to(dtype=attention_mask.dtype)
    )
    return out_ids, out_attn


def can_use_kv_cache(
    *,
    model: object,
    tokenizer: PaddingTokenizerLike,
    inputs: dict[str, torch.Tensor],
) -> tuple[bool, dict[str, torch.Tensor]]:
    del model
    normalized = dict(inputs)
    attn = normalized.get("attention_mask", None)
    if torch.is_tensor(attn) and attn.numel() > 0:
        if is_all_ones_mask(attn):
            normalized.pop("attention_mask", None)
        else:
            is_padding, is_left_padded, _lens = is_padding_mask(attn)
            if not is_padding:
                raise ValueError(
                    "attention_mask must be all-ones or a contiguous padding mask"
                )
            if not is_left_padded:
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is None:
                    pad_id = 0
                normalized["input_ids"], normalized["attention_mask"] = right_pad_to_left_pad(
                    input_ids=normalized["input_ids"],
                    attention_mask=attn,
                    pad_token_id=int(pad_id),
                )
    return True, normalized


__all__ = ["PaddingTokenizerLike", "can_use_kv_cache", "is_padding_mask", "right_pad_to_left_pad"]
