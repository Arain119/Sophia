from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic


_ABNORMAL_PATTERNS = (
    re.compile(r"<(?:message|source|translation|location)\b", re.IGNORECASE),
    re.compile(r"filename\s*=", re.IGNORECASE),
    re.compile(r"(?:^|[\s\"'])\.\.?/(?:[^\s/]+/){2,}"),
    re.compile(r"(?:^|[\s\"'])(?:org|com|src|usr|var)/(?:[^\s/]+/){2,}"),
)


@dataclass(frozen=True)
class GenerationCase:
    case_id: str
    language: str
    capability: str
    raw_prompt: str
    chat_prompt: str
    raw_expected_any: tuple[str, ...]
    chat_expected_any: tuple[str, ...]
    raw_match_mode: str
    chat_match_mode: str
    language_match_mode: str

    def prompt(self, prompt_format: str) -> str:
        return self.raw_prompt if str(prompt_format) == "raw" else self.chat_prompt

    def expected_any(self, prompt_format: str) -> tuple[str, ...]:
        return (
            self.raw_expected_any
            if str(prompt_format) == "raw"
            else self.chat_expected_any
        )

    def match_mode(self, prompt_format: str) -> str:
        return self.raw_match_mode if str(prompt_format) == "raw" else self.chat_match_mode


def _normalized_text(value: str) -> str:
    return "".join(str(value).casefold().split())


def repeated_ngram_ratio(text: str, *, n: int = 4) -> float:
    tokens = re.findall(r"[\w]+|[^\w\s]", str(text), flags=re.UNICODE)
    if len(tokens) < int(n):
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - (float(len(set(ngrams))) / float(len(ngrams)))


def response_language_match(
    *,
    response: str,
    expected_any: Iterable[str],
    language: str,
    mode: str = "auto",
) -> bool | None:
    if str(mode) == "exempt":
        return None
    if str(mode) != "auto":
        raise ValueError(f"unsupported language_match_mode: {mode!r}")
    stripped = str(response).strip()
    if not stripped:
        return False
    cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", stripped))
    latin_words = re.findall(r"[A-Za-z]+", stripped)
    target = str(language).casefold()
    if target == "en":
        return cjk_count == 0
    if target != "zh":
        return None
    if cjk_count > 0:
        return True
    if not latin_words:
        return True
    normalized_response = _normalized_text(stripped).strip(".,;:!?。；：！？")
    normalized_expected = {
        _normalized_text(item).strip(".,;:!?。；：！？")
        for item in expected_any
        if str(item).strip()
    }
    return normalized_response in normalized_expected


def analyze_response(
    *,
    prompt: str,
    response: str,
    expected_any: Iterable[str],
    match_mode: str = "contains",
    language: str = "unknown",
    language_match_mode: str = "auto",
) -> dict[str, object]:
    stripped = str(response).strip()
    normalized_response = _normalized_text(stripped)
    normalized_prompt = _normalized_text(prompt)
    expected_values = tuple(str(value) for value in expected_any)
    normalized_expected = tuple(
        item for item in (_normalized_text(value) for value in expected_values) if item
    )
    if str(match_mode) == "prefix":
        answer_match = bool(normalized_expected) and any(
            normalized_response.startswith(expected) for expected in normalized_expected
        )
    elif str(match_mode) == "contains":
        answer_match = bool(normalized_expected) and any(
            expected in normalized_response for expected in normalized_expected
        )
    else:
        raise ValueError(f"unsupported match_mode: {match_mode!r}")
    prompt_echo = bool(normalized_prompt) and (
        normalized_response == normalized_prompt
        or normalized_response.startswith(normalized_prompt)
    )
    abnormal_markup_or_path = any(
        pattern.search(stripped) is not None for pattern in _ABNORMAL_PATTERNS
    )
    repetition_ratio = repeated_ngram_ratio(stripped)
    return {
        "answer_match": bool(answer_match),
        "empty_response": not bool(stripped),
        "prompt_echo": bool(prompt_echo),
        "abnormal_markup_or_path": bool(abnormal_markup_or_path),
        "repeated_4gram_ratio": float(repetition_ratio),
        "high_repetition": bool(repetition_ratio >= 0.25),
        "language_match": response_language_match(
            response=stripped,
            expected_any=expected_values,
            language=str(language),
            mode=str(language_match_mode),
        ),
    }


def load_suite(path: str) -> list[GenerationCase]:
    cases: list[GenerationCase] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"suite row must be an object: {path}:{line_number}")
            case_id = str(payload.get("id") or "").strip()
            if not case_id or case_id in seen_ids:
                raise ValueError(f"suite id must be non-empty and unique: {case_id!r}")
            seen_ids.add(case_id)
            raw_prompt = str(payload.get("raw_prompt") or "")
            chat_prompt = str(payload.get("chat_prompt") or "")
            raw_expected = payload.get("raw_expected_any", payload.get("expected_any"))
            chat_expected = payload.get("chat_expected_any", payload.get("expected_any"))
            if not raw_prompt or not chat_prompt:
                raise ValueError(f"suite prompts must be non-empty: {case_id}")
            for label, expected in (
                ("raw_expected_any", raw_expected),
                ("chat_expected_any", chat_expected),
            ):
                if not isinstance(expected, list) or not expected or not all(
                    isinstance(item, str) and item.strip() for item in expected
                ):
                    raise ValueError(
                        f"{label} must be a non-empty string list: {case_id}"
                    )
            raw_match_mode = str(payload.get("raw_match_mode") or "prefix")
            chat_match_mode = str(payload.get("chat_match_mode") or "prefix")
            if raw_match_mode not in {"prefix", "contains"} or chat_match_mode not in {
                "prefix",
                "contains",
            }:
                raise ValueError(f"invalid match mode: {case_id}")
            language_match_mode = str(
                payload.get("language_match_mode") or "auto"
            )
            if language_match_mode not in {"auto", "exempt"}:
                raise ValueError(f"invalid language match mode: {case_id}")
            cases.append(
                GenerationCase(
                    case_id=case_id,
                    language=str(payload.get("language") or "unknown"),
                    capability=str(payload.get("capability") or "unknown"),
                    raw_prompt=raw_prompt,
                    chat_prompt=chat_prompt,
                    raw_expected_any=tuple(str(item) for item in raw_expected),
                    chat_expected_any=tuple(str(item) for item in chat_expected),
                    raw_match_mode=raw_match_mode,
                    chat_match_mode=chat_match_mode,
                    language_match_mode=language_match_mode,
                )
            )
    if not cases:
        raise ValueError(f"generation suite is empty: {path}")
    return cases


def select_cases(
    cases: list[GenerationCase],
    *,
    max_cases: int | None,
    seed: int = 2026,
) -> list[GenerationCase]:
    """Select a deterministic, language/capability-balanced diagnostic subset.

    Release evaluation keeps ``max_cases=None`` and therefore evaluates every
    pinned case.  Smaller subsets are for checkpoint trajectory checks only;
    hashing the case id makes the selection independent of JSONL ordering.
    """
    if max_cases is None or int(max_cases) <= 0 or int(max_cases) >= len(cases):
        return list(cases)
    target = int(max_cases)
    groups: dict[tuple[str, str], list[GenerationCase]] = defaultdict(list)
    for case in cases:
        groups[(case.language, case.capability)].append(case)
    ordered_groups = sorted(groups.items())
    ranked: dict[tuple[str, str], list[GenerationCase]] = {}
    for group, members in ordered_groups:
        ranked[group] = sorted(
            members,
            key=lambda case: hashlib.sha256(
                f"{int(seed)}:{case.case_id}".encode()
            ).digest(),
        )
    selected: list[GenerationCase] = []
    pointers = {group: 0 for group, _members in ordered_groups}
    # Round-robin strata so a small diagnostic still represents every domain,
    # while never repeating a case when a stratum is exhausted.
    while len(selected) < target:
        progressed = False
        for group, _members in ordered_groups:
            index = pointers[group]
            members = ranked[group]
            if index >= len(members):
                continue
            selected.append(members[index])
            pointers[group] = index + 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            raise RuntimeError("case selection exhausted before reaching max_cases")
    return sorted(selected, key=lambda case: case.case_id)


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, object]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, object]:
        count = len(group)
        denominator = float(max(count, 1))
        language_rows = [
            row for row in group if row.get("language_match") is not None
        ]
        language_denominator = float(max(len(language_rows), 1))
        return {
            "cases": int(count),
            "answer_accuracy": sum(bool(row["answer_match"]) for row in group)
            / denominator,
            "empty_rate": sum(bool(row["empty_response"]) for row in group)
            / denominator,
            "prompt_echo_rate": sum(bool(row["prompt_echo"]) for row in group)
            / denominator,
            "abnormal_markup_or_path_rate": sum(
                bool(row["abnormal_markup_or_path"]) for row in group
            )
            / denominator,
            "high_repetition_rate": sum(bool(row["high_repetition"]) for row in group)
            / denominator,
            "first_token_eos_rate": sum(
                int(row.get("first_token_top1_id", -1))
                == int(row.get("eos_token_id", -2))
                for row in group
            )
            / denominator,
            "first_token_eos_probability_mean": sum(
                float(row.get("first_token_eos_probability", 0.0)) for row in group
            )
            / denominator,
            "language_match_cases": len(language_rows),
            "language_match_rate": sum(
                bool(row.get("language_match")) for row in language_rows
            )
            / language_denominator,
        }

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row["language"])].append(row)
        by_capability[str(row["capability"])].append(row)
    return {
        "overall": summarize(rows),
        "by_language": {
            key: summarize(value) for key, value in sorted(by_language.items())
        },
        "by_capability": {
            key: summarize(value) for key, value in sorted(by_capability.items())
        },
    }


def _raw_inputs(tokenizer: object, prompt: str) -> dict[str, object]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing evaluation binding file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_binding(
    *,
    protocol_json: str | Path | None,
    checkpoint: str | Path | None,
    checkpoint_sha256: str | None,
    eval_export_model: str | Path | None,
    evaluation_asset_sha256: str,
) -> dict[str, str | None] | None:
    values = (protocol_json, checkpoint, checkpoint_sha256, eval_export_model)
    if not any(value is not None and str(value).strip() for value in values):
        return None
    if not all(value is not None and str(value).strip() for value in (protocol_json, checkpoint, eval_export_model)):
        raise ValueError(
            "release-bound evaluation requires --protocol_json, --checkpoint, and --eval_export_model"
        )
    protocol_path = Path(str(protocol_json)).expanduser().resolve()
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    model_path = Path(str(eval_export_model)).expanduser().resolve()
    protocol_digest = _sha256(protocol_path)
    checkpoint_digest = _sha256(checkpoint_path)
    model_digest = _sha256(model_path)
    if checkpoint_sha256 and str(checkpoint_sha256).lower() != checkpoint_digest:
        raise ValueError("checkpoint_sha256 does not match checkpoint")
    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "eval_export_model_path": str(model_path),
        "eval_export_model_sha256": model_digest,
        "evaluation_asset_sha256": str(evaluation_asset_sha256),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    from ml.modeling.decoder_output import require_output_logits
    from ml.runtime.controls import maybe_reset_runtime_cache
    from ml.runtime.inference.eval.generation_padding import can_use_kv_cache
    from ml.runtime.inference.eval.generation_runtime import generate_with_kv_cache
    from ml.runtime.inference.eval.model_io import init_model
    from ml.runtime.inference.eval.runtime import (
        build_inputs_from_conversation,
        clamp_generation_window,
    )
    from ml.runtime.inference.eval.runtime_config import (
        EvalRuntimeConfig,
        resolve_eval_runtime_config,
    )
    from ml.runtime.inference.eval.common import torch

    if str(args.kda_backend) == "reference" and any(
        str(value or "").strip()
        for value in (args.protocol_json, args.checkpoint, args.eval_export_model)
    ):
        raise ValueError(
            "reference KDA backend is diagnostic-only and cannot create release-bound evidence"
        )
    runtime_config = resolve_eval_runtime_config(
        EvalRuntimeConfig(
            export_dir=str(args.export_dir),
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            do_sample=bool(int(args.do_sample)),
            top_k=int(args.top_k),
            use_cache=True,
            device=str(args.device),
            mode="auto",
        )
    )
    suite_path = Path(str(args.suite)).expanduser().resolve()
    suite_sha256 = _sha256(suite_path)
    all_cases = load_suite(str(suite_path))
    cases = select_cases(
        all_cases,
        max_cases=(None if int(args.max_cases) <= 0 else int(args.max_cases)),
        seed=int(args.selection_seed),
    )
    model, tokenizer = init_model(runtime_config)
    if str(args.kda_backend) == "reference":
        for module in model.modules():
            if hasattr(module, "backend"):
                module.backend = "reference"
    device = torch.device(str(runtime_config.device))
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        prompt = case.prompt(str(args.prompt_format))
        expected_any = case.expected_any(str(args.prompt_format))
        match_mode = case.match_mode(str(args.prompt_format))
        inputs = (
            _raw_inputs(tokenizer, prompt)
            if str(args.prompt_format) == "raw"
            else build_inputs_from_conversation(
                tokenizer=tokenizer,
                conversation=[{"role": "user", "content": prompt}],
            )
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        _can_cache, inputs = can_use_kv_cache(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
        )
        inputs, max_new_tokens, cache_len = clamp_generation_window(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=int(runtime_config.max_new_tokens),
            max_cache_len=0,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        maybe_reset_runtime_cache(model)
        with torch.inference_mode():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
                logits_to_keep=1,
            )
            first_logits = require_output_logits(
                output,
                context="generation quality first-token evaluation",
            )[0, -1].float()
            first_probs = torch.softmax(first_logits, dim=-1)
            top1_id = int(torch.argmax(first_logits).item())
            eos_id = tokenizer.eos_token_id
            eos_probability = (
                0.0 if eos_id is None else float(first_probs[int(eos_id)].item())
            )
            generated = generate_with_kv_cache(
                model=model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(runtime_config.do_sample),
                temperature=float(runtime_config.temperature),
                top_p=float(runtime_config.top_p),
                top_k=int(runtime_config.top_k),
                eos_token_id=eos_id,
                max_cache_len=int(cache_len),
            )
        generated_tokens = generated[0, prompt_len:].detach().cpu().tolist()
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        analysis = analyze_response(
            prompt=prompt,
            response=response,
            expected_any=expected_any,
            match_mode=match_mode,
            language=case.language,
            language_match_mode=case.language_match_mode,
        )
        row = {
            "id": case.case_id,
            "language": case.language,
            "capability": case.capability,
            "prompt": prompt,
            "expected_any": list(expected_any),
            "match_mode": match_mode,
            "response": response,
            "generated_tokens": len(generated_tokens),
            "eos_token_id": None if eos_id is None else int(eos_id),
            "first_token_top1_id": int(top1_id),
            "first_token_top1_text": tokenizer.decode([top1_id]),
            "first_token_eos_probability": float(eos_probability),
            **analysis,
        }
        rows.append(row)
        print(
            f"[EVAL] {index}/{len(cases)} id={case.case_id} "
            f"match={int(bool(row['answer_match']))} "
            f"eos_p={eos_probability:.4f} response={response[:100]!r}",
            flush=True,
        )
    report = {
        "schema": "sophia_generation_quality_v1",
        "model_dir": os.path.abspath(str(args.export_dir)),
        "suite_path": str(suite_path),
        "suite_sha256": suite_sha256,
        "selection": {
            "full_case_count": len(all_cases),
            "evaluated_case_count": len(cases),
            "max_cases": int(args.max_cases),
            "seed": int(args.selection_seed),
            "balanced_by": ["language", "capability"],
            "case_ids": [case.case_id for case in cases],
        },
        "evaluation_binding": _evaluation_binding(
            protocol_json=args.protocol_json,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            eval_export_model=args.eval_export_model,
            evaluation_asset_sha256=suite_sha256,
        ),
        "prompt_format": str(args.prompt_format),
        "decode": {
            "seed": int(runtime_config.seed),
            "max_new_tokens": int(runtime_config.max_new_tokens),
            "do_sample": bool(runtime_config.do_sample),
            "temperature": float(runtime_config.temperature),
            "top_p": float(runtime_config.top_p),
            "top_k": int(runtime_config.top_k),
        },
        "runtime": {"kda_backend": str(args.kda_backend)},
        "summary": aggregate_results(rows),
        "cases": rows,
    }
    write_json_atomic(
        str(args.output),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    default_suite = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "eval"
        / "generation_quality.jsonl"
    )
    parser = argparse.ArgumentParser(description="Evaluate Sophia generation quality gates")
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default=str(default_suite))
    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="diagnostic subset size; 0 evaluates the complete pinned suite",
    )
    parser.add_argument("--selection_seed", type=int, default=2026)
    parser.add_argument("--prompt_format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", type=int, choices=(0, 1), default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument(
        "--kda_backend",
        choices=("auto", "reference"),
        default="auto",
        help="diagnostic-only KDA backend override; auto is required for release evaluation",
    )
    parser.add_argument("--protocol_json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument("--eval_export_model")
    return parser


def main() -> None:
    run_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()
