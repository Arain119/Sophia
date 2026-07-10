from __future__ import annotations

import json
import os

import pytest

import ml.modeling.text as _sophia_text_pkg
from ml.integrations.adapters.hf.tokenizer import RuntimeTokenizer, load_local_tokenizer
from ml.errors import SophiaUsageError
from ml.modeling.text.conversation import render_conversation_segments

_BUNDLE_TOKENIZER_DIR = os.path.dirname(os.path.abspath(_sophia_text_pkg.__file__))


def _write_bundle(tmp_path, *, model_max_length: int = 4096) -> str:
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tok_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "model_max_length": int(model_max_length),
                "eos_token": "<eos>",
                "pad_token": "<pad>",
            }
        ),
        encoding="utf-8",
    )
    return str(tok_dir)


def test_load_local_tokenizer_requires_local_bundle(tmp_path) -> None:
    with pytest.raises(SophiaUsageError, match="tokenizer_path must be a local directory"):
        load_local_tokenizer(str(tmp_path / "missing"))


def test_load_local_tokenizer_applies_runtime_model_max_length(monkeypatch, tmp_path) -> None:
    tok_dir = _write_bundle(tmp_path)

    class DummyTokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"
            self.truncation_side = "right"
            self.pad_token_id = None
            self.eos_token_id = 3
            self.eos_token = "<eos>"
            self.init_kwargs: dict[str, int] = {"model_max_length": 4096}
            self.model_max_length = 4096

    dummy = DummyTokenizer()

    class DummyTokenizerLoader:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return dummy

    monkeypatch.setattr(
        "ml.integrations.adapters.hf.tokenizer.PreTrainedTokenizerFast",
        DummyTokenizerLoader,
    )
    monkeypatch.setattr(
        "ml.integrations.adapters.hf.tokenizer.PreTrainedTokenizerBase",
        DummyTokenizer,
    )

    tokenizer = load_local_tokenizer(
        tok_dir,
        padding_side="left",
        truncation_side="left",
        model_max_length=4096,
    )

    assert isinstance(tokenizer, RuntimeTokenizer)
    assert tokenizer.base_tokenizer is dummy
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"
    assert tokenizer.pad_token_id == 3
    assert tokenizer.model_max_length == 4096
    assert tokenizer.init_kwargs["model_max_length"] == 4096


def test_bundle_chat_template_matches_python_renderer() -> None:
    tok = load_local_tokenizer(_BUNDLE_TOKENIZER_DIR, model_max_length=4096)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {
            "role": "user",
            "content_blocks": [
                {"type": "text", "text": "q2"},
                {"type": "text", "text": "q3"},
            ],
        },
        {"role": "assistant", "content": "a2"},
    ]
    rendered = "".join(
        segment.text
        for segment in render_conversation_segments(
            messages,
            add_generation_prompt=False,
        )
    )
    templated = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    assert templated == rendered


def test_bundle_includes_posttrain_chat_tokens() -> None:
    tok = load_local_tokenizer(_BUNDLE_TOKENIZER_DIR, model_max_length=4096)
    for token in ("<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"):
        assert token in tok.all_special_tokens
        token_id = tok.convert_tokens_to_ids(token)
        assert token_id != tok.unk_token_id
        assert tok.encode(token, add_special_tokens=False) == [int(token_id)]


@pytest.mark.parametrize(
    ("messages", "add_generation_prompt"),
    [
        (
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1", "wo_eos": True},
            ],
            True,
        ),
        (
            [
                {"role": "developer", "content": "policy"},
                {"role": "user", "content": "q1"},
            ],
            True,
        ),
        (
            [
                {"role": "system", "content": "sys", "response_format": {"type": "object", "properties": {"answer": {"type": "string"}}}},
                {"role": "user", "content": "q1"},
            ],
            True,
        ),
        (
            [
                {"role": "user", "content": "q1"},
            ],
            True,
        ),
    ],
)
def test_bundle_chat_template_matches_python_renderer_extra_cases(messages, add_generation_prompt) -> None:
    tok = load_local_tokenizer(_BUNDLE_TOKENIZER_DIR, model_max_length=4096)
    rendered = "".join(
        segment.text
        for segment in render_conversation_segments(
            messages,
            add_generation_prompt=add_generation_prompt,
        )
    )
    templated = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    assert templated == rendered
