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
    ToolCommand("posttrain", "profile", "ml.tooling.scripts.data.production.profile_posttrain_lengths:main", "Profile post-train dataset lengths."),
    ToolCommand("posttrain", "curriculum", "ml.tooling.scripts.data.production.write_posttrain_length_curriculum:main", "Write the production post-train length curriculum."),
    ToolCommand("posttrain", "export", "ml.tooling.scripts.machine_recipes.export_machine_recipe:main", "Export a normalized machine-recipe json from SFT machine-selection outputs."),
    ToolCommand("posttrain", "select", "ml.tooling.scripts.machine_recipes.select_sft_machine_recipe:main", "Select the SFT machine recipe under fixed release semantics and report the machine-layer Pareto frontier."),
    ToolCommand("pretrain", "check", "ml.tooling.scripts.machine_recipes.check_pretrain_machine_recipe:main", "Validate the fixed pretrain semantic recipe and report the machine-layer runtime selection."),
    ToolCommand("pretrain", "breakdown", "ml.tooling.scripts.pretrain_step_breakdown:main", "Break down real pretrain StepRunner micro/clip/optimizer time."),
    ToolCommand("pretrain", "gpu-sweep", "ml.tooling.scripts.pretrain_gpu_sweep:main", "Sweep pretrain batch/loss/runtime candidates on the target GPU."),
    ToolCommand("pretrain", "stability", "ml.tooling.scripts.pretrain_stability_probe:main", "Run or summarize matched pretrain stability probes before release training."),
    ToolCommand("audit", "token-shards", "ml.tooling.scripts.data.audit_token_shards:main", "Audit decoded samples from token shards."),
    ToolCommand("audit", "resume", "ml.tooling.scripts.release_gate.audit_resume_drill:main", "Audit machine-level resume drill evidence for pretrain or SFT."),
    ToolCommand("audit", "soak", "ml.tooling.scripts.release_gate.audit_soak_test:main", "Audit target-machine soak-test evidence for release training readiness."),
    ToolCommand("audit", "release", "ml.tooling.scripts.release_gate.audit_training_release:main", "Audit training release artifacts, machine recipes, and checkpoint-eval outputs against the release gate."),
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
        action_parser = domains[command.domain].add_parser(command.action, help=command.help)
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
        func()
    return 0


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
