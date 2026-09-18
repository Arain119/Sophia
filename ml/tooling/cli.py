from __future__ import annotations

import argparse
from collections.abc import Iterator
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass

from ml.cli.support import run_cli
from ml.core.common.mapping import object_mapping
from ml.runtime.stack import ensure_standard_stack


@dataclass(frozen=True)
class ToolCommand:
    domain: str
    action: str
    target: str
    help: str


_COMMANDS: tuple[ToolCommand, ...] = (
    ToolCommand(
        "sft",
        "train",
        "ml.cli.sft_train:main",
        "Run or dry-run the packed completion-only SFT trainer.",
    ),
    ToolCommand(
        "rl",
        "local-score",
        "ml.tooling.scripts.score_local_rollouts:main",
        "Attach deterministic local rewards without consuming provider budget.",
    ),
    ToolCommand(
        "rl",
        "dpo-pairs",
        "ml.tooling.scripts.build_dpo_pairs:main",
        "Select margin-separated preference pairs from judged rollout groups.",
    ),
    ToolCommand(
        "rl",
        "dpo-train",
        "ml.tooling.scripts.train_dpo:main",
        "Run a bounded reference-cached DPO pass over a SFT export.",
    ),
    ToolCommand(
        "rl",
        "rollout",
        "ml.tooling.scripts.sample_rl_rollouts:main",
        "Sample fixed-size policy rollout groups for preference or GRPO training.",
    ),
    ToolCommand(
        "rl",
        "grpo-train",
        "ml.tooling.scripts.train_grpo:main",
        "Run a bounded group-relative policy optimisation pass.",
    ),
    ToolCommand(
        "pretrain",
        "check",
        "ml.tooling.scripts.machine_recipes.check_pretrain_machine_recipe:main",
        "Validate the fixed pretrain semantic recipe and report the machine-layer runtime selection.",
    ),
    ToolCommand(
        "pretrain",
        "stability",
        "ml.tooling.scripts.pretrain_stability_probe:main",
        "Run or summarize the pretrain stability and resume probes before release training.",
    ),
    ToolCommand(
        "pretrain",
        "generation-suite",
        "ml.tooling.scripts.build_generation_quality_suite:main",
        "Build the pinned 2,000-case bilingual base generation and decontamination suite.",
    ),
    ToolCommand(
        "pretrain",
        "executable-eval",
        "ml.tooling.scripts.eval_executable_capabilities:main",
        "Run restricted AST-whitelisted code tests and programmatic math exact-match evaluation.",
    ),
    ToolCommand(
        "pretrain",
        "generation-eval",
        "ml.tooling.scripts.eval_generation_quality:main",
        "Evaluate raw base-model generation with a pinned or stratified UTF-8 suite.",
    ),
    ToolCommand(
        "pretrain",
        "perplexity-eval",
        "ml.tooling.scripts.eval_pretrain_perplexity:main",
        "Evaluate held-out token loss and perplexity for one or more manifests.",
    ),
    ToolCommand(
        "pretrain",
        "eval-trajectory",
        "ml.tooling.scripts.summarize_pretrain_eval_trajectory:main",
        "Combine checkpoint evaluation reports into a comparable diagnostic trajectory.",
    ),
    ToolCommand(
        "pretrain",
        "final-check",
        "ml.tooling.scripts.eval_pretrain_base_capability_gate:main",
        "Validate the final seed-42 checkpoint and its bound evaluation evidence.",
    ),
    ToolCommand(
        "pretrain",
        "lifecycle-validate",
        "ml.tooling.scripts.validate_model_lifecycle:main",
        "Validate scaled forward/backward, checkpoint/resume, and export/reload semantics on CUDA.",
    ),
    ToolCommand(
        "pretrain",
        "tokenizer-train",
        "ml.tooling.scripts.data.train_tokenizer:main",
        "Train a bounded tokenizer and write input-file provenance.",
    ),
    ToolCommand(
        "pretrain",
        "tokenizer-code-corpus",
        "ml.tooling.scripts.data.build_tokenizer_code_corpus:main",
        "Build a provenance-rich code and structured-text corpus from pinned archives.",
    ),
    ToolCommand(
        "pretrain",
        "source-inventory",
        "ml.tooling.scripts.data.snapshot_pretrain_source_inventory:main",
        "Snapshot pinned, admitted Hugging Face files for the new pretrain corpus.",
    ),
    ToolCommand(
        "pretrain",
        "source-download",
        "ml.tooling.scripts.data.download_hf_source_inventory:main",
        "Download and hash-verify a pinned Hugging Face source inventory.",
    ),
    ToolCommand(
        "pretrain",
        "wikimedia-inventory",
        "ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory:main",
        "Snapshot immutable, checksum-pinned Wikimedia article dumps.",
    ),
    ToolCommand(
        "pretrain",
        "wikimedia-select",
        "ml.tooling.scripts.data.select_wikimedia_dump_subset:main",
        "Select complete, evenly spaced Wikimedia article/index pairs from a pinned dump inventory.",
    ),
    ToolCommand(
        "pretrain",
        "wikimedia-download",
        "ml.tooling.scripts.data.download_wikimedia_dump_inventory:main",
        "Download and verify a pinned Wikimedia dump inventory.",
    ),
    ToolCommand(
        "pretrain",
        "wikimedia-extract",
        "ml.tooling.scripts.data.extract_wikimedia_dump:main",
        "Extract provenance-rich text from pinned Wikimedia article dumps.",
    ),
    ToolCommand(
        "pretrain",
        "wikimedia-profile",
        "ml.tooling.scripts.data.profile_wikimedia_extraction:main",
        "Profile revision-time coverage and script normalization in extracted Wikimedia text.",
    ),
    ToolCommand(
        "pretrain",
        "acquisition-bundle",
        "ml.tooling.scripts.data.build_pretrain_acquisition_bundle:main",
        "Merge verified HF and Wikimedia acquisitions for one global clean run.",
    ),
    ToolCommand(
        "pretrain",
        "acquisition-extend",
        "ml.tooling.scripts.data.extend_pretrain_acquisition_bundle:main",
        "Append verified sources and derivatives to a frozen acquisition lineage.",
    ),
    ToolCommand(
        "pretrain",
        "humanities-filter",
        "ml.tooling.scripts.data.filter_chinese_humanities_derivative:main",
        "Build a quality- and taxonomy-filtered Chinese humanities derivative.",
    ),
    ToolCommand(
        "pretrain",
        "legacy-content-exclusion",
        "ml.tooling.scripts.data.build_legacy_content_exclusion:main",
        "Build a resumable exact and near-content exclusion index from legacy Sophia corpora.",
    ),
    ToolCommand(
        "pretrain",
        "quality-proxy-train",
        "ml.tooling.scripts.data.train_chinese_quality_proxy:main",
        "Train and freeze a lightweight Chinese educational-quality proxy from pinned human labels.",
    ),
    ToolCommand(
        "pretrain",
        "chinese-candidate-profile",
        "ml.tooling.scripts.data.profile_modern_chinese_candidate:main",
        "Profile distributed quality, script, domain, and legacy overlap for a Chinese candidate.",
    ),
    ToolCommand(
        "pretrain",
        "benchmark-decontamination",
        "ml.tooling.scripts.data.build_pretrain_decontamination_benchmarks:main",
        "Acquire pinned evaluation suites and build merged pretrain decontamination signatures.",
    ),
    ToolCommand(
        "pretrain",
        "corpus-clean",
        "ml.tooling.scripts.data.clean_pretrain_corpus:main",
        "Clean, license-filter, deduplicate, decontaminate, and split the new pretrain corpus.",
    ),
    ToolCommand(
        "pretrain",
        "corpus-profile",
        "ml.tooling.scripts.data.profile_pretrain_corpus:main",
        "Measure exact tokenizer-level source and long-context token supply in the cleaned corpus.",
    ),
    ToolCommand(
        "pretrain",
        "mix-materialize",
        "ml.tooling.scripts.data.materialize_pretrain_mix_policy:main",
        "Materialize a resolver-compatible pretrain mix from measured source supply.",
    ),
    ToolCommand(
        "pretrain",
        "mix-resolve",
        "ml.tooling.scripts.data.resolve_pretrain_mix_plan:main",
        "Resolve unique-only source and length-bucket quotas from the measured corpus supply.",
    ),
    ToolCommand(
        "pretrain",
        "shard-build",
        "ml.tooling.scripts.data.build_pretrain_token_shards:main",
        "Build unique, provenance-complete token shards from the resolved pretrain mix.",
    ),
    ToolCommand(
        "audit",
        "token-shards",
        "ml.tooling.scripts.data.audit_token_shards:main",
        "Audit decoded samples from token shards.",
    ),
    ToolCommand(
        "audit",
        "tokenizer",
        "ml.tooling.scripts.data.audit_tokenizer_quality:main",
        "Audit tokenizer fertility, fallback, roundtrip behavior, protocol, and throughput.",
    ),
    ToolCommand(
        "audit",
        "tokenizer-heldout-suite",
        "ml.tooling.scripts.data.build_tokenizer_shard_heldout_suite:main",
        "Build a deterministic tokenizer audit suite from held-out token shards.",
    ),
    ToolCommand(
        "audit",
        "tokenizer-gates",
        "ml.tooling.scripts.eval_tokenizer_quality_gates:main",
        "Compare a tokenizer candidate with a baseline using frozen quality gates.",
    ),
    ToolCommand(
        "audit",
        "resume",
        "ml.tooling.scripts.release_gate.audit_resume_drill:main",
        "Audit machine-level resume drill evidence for pretraining.",
    ),
    ToolCommand(
        "audit",
        "soak",
        "ml.tooling.scripts.release_gate.audit_soak_test:main",
        "Audit target-machine soak-test evidence for release training readiness.",
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-tool",
        description="Unified production tooling entrypoint for ML maintenance workflows.",
    )
    domain_subparsers = parser.add_subparsers(dest="domain")
    domains: dict[str, argparse._SubParsersAction[argparse.ArgumentParser]] = {}
    for command in _COMMANDS:
        if command.domain not in domains:
            domain_parser = domain_subparsers.add_parser(command.domain)
            domains[command.domain] = domain_parser.add_subparsers(dest="action")
        action_parser = domains[command.domain].add_parser(
            command.action, help=command.help
        )
        action_parser.set_defaults(target=command.target)
    return parser


@contextmanager
def _temporary_argv(argv: list[str]) -> Iterator[None]:
    previous_argv = list(sys.argv)
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous_argv


def _run_target(target: str, argv: list[str], *, prog: str) -> int:
    ensure_standard_stack(mode="tools")
    module_name, func_name = str(target).split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    with _temporary_argv([prog, *argv]):
        result = func()
    return int(result) if isinstance(result, int) else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parsed, remainder = parser.parse_known_args(raw_argv)
    payload = object_mapping(parsed)
    target = str(payload.get("target", "") or "")
    domain = str(payload.get("domain", "") or "")
    action = str(payload.get("action", "") or "")
    if not target:
        parser.print_help()
        return 2
    prog = " ".join(part for part in (parser.prog, domain, action) if part).strip()
    return _run_target(str(target), remainder, prog=prog)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
