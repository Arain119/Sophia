from __future__ import annotations

from ml.runtime.inference.eval.common import (
    encode_conversation,
    torch,
)
from ml.runtime.inference.eval.generation_padding import can_use_kv_cache
from ml.runtime.inference.eval.generation_runtime import generate_with_kv_cache
from ml.runtime.inference.eval.model_io import init_model, resolve_max_pos
from ml.runtime.inference.eval.runtime_config import EvalRuntimeConfig
from ml.runtime.inference.contracts import InferenceTokenizer


DEFAULT_PROMPTS = [
    "Summarize your capabilities in one paragraph.",
    "Why is the sky blue?",
    "Write a Python function that returns Fibonacci numbers.",
    "Explain photosynthesis in simple terms.",
    "What should I bring if it might rain tomorrow?",
    "Compare cats and dogs as pets.",
    "Explain what machine learning is.",
    "Recommend a few Chinese dishes to try.",
]


def build_inputs_from_conversation(
    *,
    tokenizer: InferenceTokenizer,
    conversation: list[dict[str, str]],
) -> dict[str, torch.Tensor]:
    encoded = encode_conversation(
        tokenizer=tokenizer,
        messages=list(conversation),
        max_seq_len=int(getattr(tokenizer, "model_max_length", 0) or 0),
        add_generation_prompt=True,
    )
    return {
        "input_ids": encoded.input_ids.unsqueeze(0),
        "attention_mask": encoded.attention_mask.unsqueeze(0),
    }

def clamp_generation_window(
    *,
    model: object,
    tokenizer: object,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    max_cache_len: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    del tokenizer
    max_pos = resolve_max_pos(model)
    prompt_len = int(inputs["input_ids"].shape[1])
    clamped_new_tokens = int(max_new_tokens)

    if prompt_len >= int(max_pos):
        keep = max(int(max_pos) - 1, 1)
        inputs["input_ids"] = inputs["input_ids"][:, -keep:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -keep:]
        prompt_len = keep
    allow_new = int(max_pos) - int(prompt_len)
    clamped_new_tokens = max(0, min(int(clamped_new_tokens), int(allow_new)))

    cache_len = int(max_cache_len)
    if cache_len <= 0:
        cache_len = int(max_pos)
    if cache_len > 0:
        allow_new_cache = int(cache_len) - int(prompt_len)
        clamped_new_tokens = max(0, min(int(clamped_new_tokens), int(allow_new_cache)))

    return inputs, int(clamped_new_tokens), int(cache_len)


def run(config: EvalRuntimeConfig) -> None:
    conversation: list[dict[str, str]] = []
    model, tokenizer = init_model(config)

    if config.mode == "ask":
        try:
            input_mode = int(input("[0] auto prompts\n[1] manual input\n"))
        except ValueError:
            input_mode = 0
    elif config.mode == "auto":
        input_mode = 0
    else:
        input_mode = 1

    prompt_iter = DEFAULT_PROMPTS if input_mode == 0 else iter(lambda: input("User: "), "")
    for prompt in prompt_iter:
        if input_mode == 0:
            print(f"User: {prompt}")

        conversation = (
            conversation[-int(config.history_turns) :] if config.history_turns else []
        )
        conversation.append({"role": "user", "content": prompt})

        inputs = build_inputs_from_conversation(
            tokenizer=tokenizer,
            conversation=conversation,
        )
        _can_use_cache, inputs = can_use_kv_cache(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
        )
        inputs, max_new_tokens, cache_len = clamp_generation_window(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=int(config.max_new_tokens),
            max_cache_len=int(config.max_cache_len),
        )
        prompt_len = int(inputs["input_ids"].shape[1])

        print("Assistant: ", end="")
        with torch.inference_mode():
            generated_ids = generate_with_kv_cache(
                model=model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(config.do_sample),
                temperature=float(config.temperature),
                top_p=float(config.top_p),
                top_k=int(config.top_k),
                eos_token_id=tokenizer.eos_token_id,
                max_cache_len=int(cache_len),
            )

        out = generated_ids[0].tolist()
        gen = out[prompt_len:]
        # Public Sophia output intentionally includes the model's reasoning
        # block.  Remove only the terminal EOS marker, not <think> tags.
        response = tokenizer.decode(gen, skip_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            eos_text = tokenizer.decode(
                [int(tokenizer.eos_token_id)], skip_special_tokens=False
            )
            if eos_text and response.endswith(eos_text):
                response = response[: -len(eos_text)]
        response = response.strip()
        conversation.append({"role": "assistant", "content": response})
        print(response, end="")
        print("\n")


__all__ = [
    "DEFAULT_PROMPTS",
    "EvalRuntimeConfig",
    "build_inputs_from_conversation",
    "clamp_generation_window",
    "run",
]
