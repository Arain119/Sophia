from __future__ import annotations

from ml.tooling.scripts.data.sft_dialog_style import (
    judge_humanlike_chat_style,
    judge_paragraph_style,
)


def test_judge_paragraph_style_accepts_paragraph_response() -> None:
    decision = judge_paragraph_style(
        "The short answer is yes, but the useful part is how you pace it.\n\n"
        "If the model answers in one tight paragraph first and then adds a second paragraph for nuance, "
        "the result usually feels more natural and less templated."
    )
    assert decision.keep is True
    assert decision.reason == "ok"
    assert decision.paragraphs == 2


def test_judge_paragraph_style_rejects_list_heavy_response() -> None:
    decision = judge_paragraph_style(
        "Here are the steps:\n"
        "1. Open the project.\n"
        "2. Check the config.\n"
        "3. Rebuild the dataset.\n"
        "4. Validate the output."
    )
    assert decision.keep is False
    assert decision.reason == "list_heavy"


def test_judge_paragraph_style_rejects_inline_numbered_list() -> None:
    decision = judge_paragraph_style(
        "You can check it in one pass. 1. Open the admin page. 2. Review the section settings. "
        "3. Save and refresh so the hover image behavior updates."
    )
    assert decision.keep is False
    assert decision.reason == "inline_list_heavy"


def test_judge_paragraph_style_rejects_code_fence_when_disabled() -> None:
    decision = judge_paragraph_style(
        "Use this snippet if you need a tiny example, but keep the answer in prose when you can.\n\n```python\nprint('hello')\n```",
        allow_code_fences=False,
    )
    assert decision.keep is False
    assert decision.reason == "code_fence"


def test_judge_humanlike_chat_style_accepts_short_natural_reply() -> None:
    decision = judge_humanlike_chat_style(
        "我觉得不一定要改成很长。你先把最关键的那一层做好，剩下的可以边跑边补。"
    )
    assert decision.keep is True
    assert decision.reason == "ok"


def test_judge_humanlike_chat_style_rejects_botish_reply() -> None:
    decision = judge_humanlike_chat_style(
        "Certainly! Here are three suggestions to optimize your workflow. 1. Review the data. 2. Adjust the prompt. 3. Re-run the task."
    )
    assert decision.keep is False
