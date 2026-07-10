from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from ml.tooling.scripts.pretrain_step_breakdown import StepBreakdownCase, benchmark_case
from ml.training.pretrain.release_config import RELEASE_PRETRAIN_DEFAULTS


DEFAULT_BATCH_SIZES = "2,3,4,5,6,7,8,10,12,16"
DEFAULT_LOSS_CHUNKS = "0,2048,1024,512,256"
DEFAULT_MUON_ORTHO_BATCH_MAXES = "1,8,0"
DEFAULT_TARGET_TOKENS_PER_UPDATE = int(RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update)
MUON_ORTHO_BATCH_MAX_ENV = "SOPHIA_MUON_ORTHO_BATCH_MAX"
DEFAULT_REFINE_TOP_K = 4
DEFAULT_BATCH_EXPAND_FACTOR = 1.5
DEFAULT_COMPILE_MODES = "default,reduce-overhead,max-autotune-no-cudagraphs"
DEFAULT_COMPILE_REFINE_TOP_K = 2


def _parse_int_csv(value: str) -> tuple[int, ...]:
    out: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("expected at least one integer candidate")
    return tuple(out)


def _parse_str_csv(value: str) -> tuple[str, ...]:
    out: list[str] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    if not out:
        raise ValueError("expected at least one string candidate")
    return tuple(out)


def _accumulation_for_target(
    *,
    batch_size: int,
    seq_len: int,
    target_tokens_per_update: int,
) -> int:
    denom = max(int(batch_size) * int(seq_len), 1)
    return max(int(math.ceil(float(target_tokens_per_update) / float(denom))), 1)


def _device_total_memory_gb(device: str) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    cuda_device = torch.device(str(device))
    props = torch.cuda.get_device_properties(cuda_device)
    return float(props.total_memory) / float(1024**3)


def _memory_utilization(
    row: dict[str, object],
    *,
    total_memory_gb: float,
    metric: str = "peak_gb",
) -> float:
    if total_memory_gb <= 0.0:
        return 0.0
    return float(row.get(str(metric), row.get("peak_gb", 0.0)) or 0.0) / float(
        total_memory_gb
    )


def _throughput_key(row: dict[str, object]) -> tuple[float, float, int]:
    return (
        float(row.get("tokens_per_s", 0.0) or 0.0),
        float(row.get("memory_utilization", 0.0) or 0.0),
        int(row.get("batch_size", 0) or 0),
    )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=_json_default))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _default_output_jsonl(output_json: str) -> str:
    output = Path(str(output_json))
    return str(output.with_suffix(f"{output.suffix}.jsonl" if output.suffix else ".jsonl"))


def _sweep_progress_payload(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    total_memory_gb: float,
    probe_batch_sizes: list[int],
    refine_batch_sizes: tuple[int, ...] = (),
    compile_modes: tuple[str, ...] = (),
    baseline_compile_mode: str = "",
    compile_refine_top_k: int = 0,
    auto_batch_limit: int = 0,
    stopped_after_oom: bool = False,
) -> dict[str, object]:
    valid_rows = [row for row in rows if bool(row.get("ok", False))]
    ranked = sorted(valid_rows, key=_throughput_key, reverse=True)
    memory_best = None
    if valid_rows:
        memory_best = max(
            valid_rows,
            key=lambda row: (
                float(row.get("memory_utilization", 0.0) or 0.0),
                float(row.get("tokens_per_s", 0.0) or 0.0),
            ),
        )
    return {
        "kind": "pretrain_gpu_sweep",
        "status": "partial",
        "updated_at_unix": float(time.time()),
        "seq_len": int(args.seq_len),
        "target_tokens_per_update": int(args.target_tokens_per_update),
        "total_memory_gb": float(total_memory_gb),
        "batch_probe_sizes": [int(batch_size) for batch_size in probe_batch_sizes],
        "refine_batch_sizes": [int(batch_size) for batch_size in refine_batch_sizes],
        "compile_modes": [str(mode) for mode in compile_modes],
        "baseline_compile_mode": str(baseline_compile_mode),
        "compile_refine_top_k": int(compile_refine_top_k),
        "auto_batch_limit": int(auto_batch_limit),
        "stopped_after_oom": bool(stopped_after_oom),
        "results": rows,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
        "memory_best": memory_best,
    }


def _persist_progress(
    *,
    args: argparse.Namespace,
    row: dict[str, object],
    rows: list[dict[str, object]],
    total_memory_gb: float,
    probe_batch_sizes: list[int],
    refine_batch_sizes: tuple[int, ...],
    compile_modes: tuple[str, ...],
    baseline_compile_mode: str,
    compile_refine_top_k: int,
    auto_batch_limit: int,
    stopped_after_oom: bool,
) -> None:
    output_jsonl = str(getattr(args, "output_jsonl", "") or "").strip()
    if output_jsonl:
        _append_jsonl(Path(output_jsonl), row)
    partial_json = str(getattr(args, "partial_json", "") or "").strip()
    if partial_json:
        payload = _sweep_progress_payload(
            args=args,
            rows=rows,
            total_memory_gb=float(total_memory_gb),
            probe_batch_sizes=probe_batch_sizes,
            refine_batch_sizes=refine_batch_sizes,
            compile_modes=compile_modes,
            baseline_compile_mode=baseline_compile_mode,
            compile_refine_top_k=int(compile_refine_top_k),
            auto_batch_limit=int(auto_batch_limit),
            stopped_after_oom=bool(stopped_after_oom),
        )
        _write_json_atomic(Path(partial_json), payload)


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _candidate_key(candidate: SweepCandidate) -> tuple[int, int, int, str, int]:
    return (
        int(candidate.batch_size),
        int(candidate.loss_chunk_size),
        int(candidate.muon_ortho_batch_max),
        str(candidate.compile_mode),
        int(candidate.disable_cudagraphs),
    )


def _resolve_auto_batch_limit(
    *,
    configured_limit: int,
    batch_sizes: tuple[int, ...],
    seq_len: int,
    target_tokens_per_update: int,
) -> int:
    explicit = int(configured_limit)
    if explicit > 0:
        return explicit
    semantic_limit = _accumulation_for_target(
        batch_size=1,
        seq_len=int(seq_len),
        target_tokens_per_update=int(target_tokens_per_update),
    )
    return max(max(batch_sizes), int(semantic_limit))


def _next_auto_batch_size(
    *,
    current: int,
    limit: int,
    expand_factor: float,
) -> int:
    factor = max(float(expand_factor), 1.01)
    grown = int(math.ceil(float(current) * factor))
    if grown <= int(current):
        grown = int(current) + 1
    return min(int(grown), int(limit))


@dataclass(frozen=True)
class SweepCandidate:
    batch_size: int
    accumulation_steps: int
    loss_chunk_size: int
    muon_ortho_batch_max: int
    compile_mode: str = "default"
    disable_cudagraphs: int = -1

    @property
    def name(self) -> str:
        name = (
            f"bs{self.batch_size}_acc{self.accumulation_steps}_"
            f"chunk{self.loss_chunk_size}_muonmb{self.muon_ortho_batch_max}"
        )
        mode = str(self.compile_mode).strip()
        if mode and mode != "default":
            name = f"{name}_compile{mode}"
        if int(self.disable_cudagraphs) >= 0:
            name = f"{name}_cudagraphs{1 - int(self.disable_cudagraphs)}"
        return name


def build_candidates(
    *,
    batch_sizes: tuple[int, ...],
    loss_chunks: tuple[int, ...],
    muon_ortho_batch_maxes: tuple[int, ...],
    seq_len: int,
    target_tokens_per_update: int,
) -> tuple[SweepCandidate, ...]:
    candidates: list[SweepCandidate] = []
    for batch_size in batch_sizes:
        accumulation_steps = _accumulation_for_target(
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            target_tokens_per_update=int(target_tokens_per_update),
        )
        for loss_chunk_size in loss_chunks:
            for muon_ortho_batch_max in muon_ortho_batch_maxes:
                candidates.append(
                    SweepCandidate(
                        batch_size=int(batch_size),
                        accumulation_steps=int(accumulation_steps),
                        loss_chunk_size=int(loss_chunk_size),
                        muon_ortho_batch_max=int(muon_ortho_batch_max),
                    )
                )
    return tuple(candidates)


def _add_unique_candidate(
    out: list[SweepCandidate],
    seen: set[tuple[int, int, int, str, int]],
    candidate: SweepCandidate,
) -> None:
    key = _candidate_key(candidate)
    if key in seen:
        return
    seen.add(key)
    out.append(candidate)


def _mark_candidate_seen(
    seen: set[tuple[int, int, int, str, int]],
    candidate: SweepCandidate,
) -> None:
    seen.add(_candidate_key(candidate))


def _candidate_for(
    *,
    batch_size: int,
    loss_chunk_size: int,
    muon_ortho_batch_max: int,
    seq_len: int,
    target_tokens_per_update: int,
    compile_mode: str,
    disable_cudagraphs: int = -1,
) -> SweepCandidate:
    return SweepCandidate(
        batch_size=int(batch_size),
        accumulation_steps=_accumulation_for_target(
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            target_tokens_per_update=int(target_tokens_per_update),
        ),
        loss_chunk_size=int(loss_chunk_size),
        muon_ortho_batch_max=int(muon_ortho_batch_max),
        compile_mode=str(compile_mode),
        disable_cudagraphs=int(disable_cudagraphs),
    )


def _select_refine_batch_sizes(
    *,
    rows: list[dict[str, object]],
    top_k: int,
) -> tuple[int, ...]:
    valid_rows = [row for row in rows if bool(row.get("ok", False))]
    if not valid_rows:
        return ()
    selected: set[int] = set()
    for row in sorted(valid_rows, key=_throughput_key, reverse=True)[: max(int(top_k), 1)]:
        selected.add(int(row.get("batch_size", 0) or 0))
    memory_best = max(
        valid_rows,
        key=lambda row: (
            float(row.get("memory_utilization", 0.0) or 0.0),
            float(row.get("tokens_per_s", 0.0) or 0.0),
        ),
    )
    selected.add(int(memory_best.get("batch_size", 0) or 0))
    selected.add(max(int(row.get("batch_size", 0) or 0) for row in valid_rows))
    return tuple(sorted(batch_size for batch_size in selected if batch_size > 0))


@contextmanager
def _temporary_env(key: str, value: object):
    previous = os.environ.get(str(key))
    os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(str(key), None)
        else:
            os.environ[str(key)] = previous


def _benchmark_candidate(
    *,
    candidate: SweepCandidate,
    args: argparse.Namespace,
) -> dict[str, object]:
    with _temporary_env(MUON_ORTHO_BATCH_MAX_ENV, candidate.muon_ortho_batch_max):
        result = benchmark_case(
            StepBreakdownCase(loss_chunk_size=candidate.loss_chunk_size),
            data_path=str(args.data_path),
            seq_len=int(args.seq_len),
            batch_size=candidate.batch_size,
            accumulation_steps=candidate.accumulation_steps,
            warmup_updates=int(args.warmup_updates),
            timed_updates=int(args.timed_updates),
            include_optimizer=bool(int(args.include_optimizer) == 1),
            device=str(args.device),
            seed=int(args.seed),
            step_execution_backend=str(args.step_execution_backend),
            compile_mode=str(candidate.compile_mode),
            disable_cudagraphs=int(candidate.disable_cudagraphs),
        )
    if not bool(result.get("ok", False)):
        raise RuntimeError(
            f"gpu sweep candidate failed: {candidate.name}: "
            f"{result.get('error', 'unknown error')}"
        )
    result["candidate_name"] = candidate.name
    result["muon_ortho_batch_max"] = candidate.muon_ortho_batch_max
    result["compile_mode"] = str(candidate.compile_mode)
    result["disable_cudagraphs"] = int(candidate.disable_cudagraphs)
    return result


def _run_candidate(
    *,
    candidate: SweepCandidate,
    args: argparse.Namespace,
    total_memory_gb: float,
    rows: list[dict[str, object]],
    seen: set[tuple[int, int, int, str, int]],
    phase: str,
    probe_batch_sizes: list[int],
    refine_batch_sizes: tuple[int, ...],
    compile_modes: tuple[str, ...],
    baseline_compile_mode: str,
    compile_refine_top_k: int,
    auto_batch_limit: int,
    stopped_after_oom: bool,
) -> dict[str, object]:
    _mark_candidate_seen(seen, candidate)
    try:
        result = _benchmark_candidate(candidate=candidate, args=args)
    except RuntimeError as exc:
        result = {
            "ok": False,
            "candidate_name": candidate.name,
            "batch_size": int(candidate.batch_size),
            "accumulation_steps": int(candidate.accumulation_steps),
            "loss_chunk_size": int(candidate.loss_chunk_size),
            "muon_ortho_batch_max": int(candidate.muon_ortho_batch_max),
            "compile_mode": str(candidate.compile_mode),
            "disable_cudagraphs": int(candidate.disable_cudagraphs),
            "phase": str(phase),
            "error": str(exc),
        }
        if not _is_oom_error(exc):
            raise
    if bool(result.get("ok", False)):
        allocated_utilization = _memory_utilization(
            result,
            total_memory_gb=float(total_memory_gb),
            metric="peak_gb",
        )
        reserved_utilization = _memory_utilization(
            result,
            total_memory_gb=float(total_memory_gb),
            metric="peak_reserved_gb",
        )
        result["allocated_memory_utilization"] = float(allocated_utilization)
        result["reserved_memory_utilization"] = float(reserved_utilization)
        result["memory_utilization"] = float(allocated_utilization)
        result["total_memory_gb"] = float(total_memory_gb)
        result["phase"] = str(phase)
    rows.append(result)
    print(json.dumps(result, sort_keys=True), flush=True)
    _persist_progress(
        args=args,
        row=result,
        rows=rows,
        total_memory_gb=float(total_memory_gb),
        probe_batch_sizes=probe_batch_sizes,
        refine_batch_sizes=refine_batch_sizes,
        compile_modes=compile_modes,
        baseline_compile_mode=str(baseline_compile_mode),
        compile_refine_top_k=int(compile_refine_top_k),
        auto_batch_limit=int(auto_batch_limit),
        stopped_after_oom=bool(stopped_after_oom),
    )
    return result


def _run_batch_probe(
    *,
    batch_size: int,
    loss_chunks: tuple[int, ...],
    muon_ortho_batch_max: int,
    args: argparse.Namespace,
    total_memory_gb: float,
    rows: list[dict[str, object]],
    seen: set[tuple[int, int, int, str, int]],
    compile_mode: str,
    probe_batch_sizes: list[int],
    refine_batch_sizes: tuple[int, ...],
    compile_modes: tuple[str, ...],
    baseline_compile_mode: str,
    compile_refine_top_k: int,
    auto_batch_limit: int,
    stopped_after_oom: bool,
) -> dict[str, object]:
    primary = _candidate_for(
        batch_size=int(batch_size),
        loss_chunk_size=int(loss_chunks[0]),
        muon_ortho_batch_max=int(muon_ortho_batch_max),
        seq_len=int(args.seq_len),
        target_tokens_per_update=int(args.target_tokens_per_update),
        compile_mode=str(compile_mode),
    )
    result = _run_candidate(
        candidate=primary,
        args=args,
        total_memory_gb=float(total_memory_gb),
        rows=rows,
        seen=seen,
        phase="batch_probe",
        probe_batch_sizes=probe_batch_sizes,
        refine_batch_sizes=refine_batch_sizes,
        compile_modes=compile_modes,
        baseline_compile_mode=baseline_compile_mode,
        compile_refine_top_k=int(compile_refine_top_k),
        auto_batch_limit=int(auto_batch_limit),
        stopped_after_oom=bool(stopped_after_oom),
    )
    if bool(result.get("ok", False)):
        return result

    for loss_chunk_size in loss_chunks[1:]:
        retry = _candidate_for(
            batch_size=int(batch_size),
            loss_chunk_size=int(loss_chunk_size),
            muon_ortho_batch_max=int(muon_ortho_batch_max),
            seq_len=int(args.seq_len),
            target_tokens_per_update=int(args.target_tokens_per_update),
            compile_mode=str(compile_mode),
        )
        result = _run_candidate(
            candidate=retry,
            args=args,
            total_memory_gb=float(total_memory_gb),
            rows=rows,
            seen=seen,
            phase="batch_probe_loss_chunk",
            probe_batch_sizes=probe_batch_sizes,
            refine_batch_sizes=refine_batch_sizes,
            compile_modes=compile_modes,
            baseline_compile_mode=baseline_compile_mode,
            compile_refine_top_k=int(compile_refine_top_k),
            auto_batch_limit=int(auto_batch_limit),
            stopped_after_oom=bool(stopped_after_oom),
        )
        if bool(result.get("ok", False)):
            return result
    return result


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    total_memory_gb = _device_total_memory_gb(str(args.device))
    batch_sizes = tuple(sorted(set(_parse_int_csv(str(args.batch_sizes)))))
    loss_chunks = _parse_int_csv(str(args.loss_chunks))
    muon_ortho_batch_maxes = _parse_int_csv(str(args.muon_ortho_batch_maxes))
    compile_modes = _parse_str_csv(
        str(getattr(args, "compile_modes", DEFAULT_COMPILE_MODES))
    )
    baseline_compile_mode = str(compile_modes[0])
    refine_top_k = max(int(getattr(args, "refine_top_k", DEFAULT_REFINE_TOP_K)), 1)
    compile_refine_top_k = max(
        int(getattr(args, "compile_refine_top_k", DEFAULT_COMPILE_REFINE_TOP_K)),
        0,
    )
    auto_expand = bool(int(getattr(args, "auto_expand_batch_sizes", 1)) == 1)
    auto_batch_limit = _resolve_auto_batch_limit(
        configured_limit=int(getattr(args, "max_auto_batch_size", 0)),
        batch_sizes=batch_sizes,
        seq_len=int(args.seq_len),
        target_tokens_per_update=int(args.target_tokens_per_update),
    )
    expand_factor = float(
        getattr(args, "batch_expand_factor", DEFAULT_BATCH_EXPAND_FACTOR)
    )
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, str, int]] = set()
    probe_batch_sizes = list(batch_sizes)
    probe_index = 0
    stopped_after_oom = False
    while probe_index < len(probe_batch_sizes):
        batch_size = int(probe_batch_sizes[probe_index])
        result = _run_batch_probe(
            batch_size=int(batch_size),
            loss_chunks=loss_chunks,
            muon_ortho_batch_max=int(muon_ortho_batch_maxes[0]),
            args=args,
            total_memory_gb=float(total_memory_gb),
            rows=rows,
            seen=seen,
            compile_mode=str(baseline_compile_mode),
            probe_batch_sizes=probe_batch_sizes,
            refine_batch_sizes=(),
            compile_modes=compile_modes,
            baseline_compile_mode=str(baseline_compile_mode),
            compile_refine_top_k=int(compile_refine_top_k),
            auto_batch_limit=int(auto_batch_limit),
            stopped_after_oom=bool(stopped_after_oom),
        )
        if not bool(result.get("ok", False)):
            stopped_after_oom = True
            break
        if (
            auto_expand
            and probe_index == len(probe_batch_sizes) - 1
            and int(batch_size) < int(auto_batch_limit)
        ):
            next_batch_size = _next_auto_batch_size(
                current=int(batch_size),
                limit=int(auto_batch_limit),
                expand_factor=float(expand_factor),
            )
            if next_batch_size not in probe_batch_sizes:
                probe_batch_sizes.append(int(next_batch_size))
        probe_index += 1

    refine_batch_sizes = _select_refine_batch_sizes(rows=rows, top_k=int(refine_top_k))
    refine_candidates: list[SweepCandidate] = []
    refine_seen: set[tuple[int, int, int]] = set()
    for batch_size in refine_batch_sizes:
        for loss_chunk_size in loss_chunks:
            for muon_ortho_batch_max in muon_ortho_batch_maxes:
                candidate = _candidate_for(
                    batch_size=int(batch_size),
                    loss_chunk_size=int(loss_chunk_size),
                    muon_ortho_batch_max=int(muon_ortho_batch_max),
                    seq_len=int(args.seq_len),
                    target_tokens_per_update=int(args.target_tokens_per_update),
                    compile_mode=str(baseline_compile_mode),
                )
                if _candidate_key(candidate) not in seen:
                    _add_unique_candidate(refine_candidates, refine_seen, candidate)

    for candidate in refine_candidates:
        _run_candidate(
            candidate=candidate,
            args=args,
            total_memory_gb=float(total_memory_gb),
            rows=rows,
            seen=seen,
            phase="refine",
            probe_batch_sizes=probe_batch_sizes,
            refine_batch_sizes=refine_batch_sizes,
            compile_modes=compile_modes,
            baseline_compile_mode=str(baseline_compile_mode),
            compile_refine_top_k=int(compile_refine_top_k),
            auto_batch_limit=int(auto_batch_limit),
            stopped_after_oom=bool(stopped_after_oom),
        )

    compile_candidates: list[SweepCandidate] = []
    compile_seen: set[tuple[int, int, int, str, int]] = set()
    if compile_refine_top_k > 0 and len(compile_modes) > 1:
        compile_base_rows = sorted(
            [row for row in rows if bool(row.get("ok", False))],
            key=_throughput_key,
            reverse=True,
        )[: int(compile_refine_top_k)]
        for row in compile_base_rows:
            for compile_mode in compile_modes[1:]:
                candidate = _candidate_for(
                    batch_size=int(row.get("batch_size", 0) or 0),
                    loss_chunk_size=int(row.get("loss_chunk_size", 0) or 0),
                    muon_ortho_batch_max=int(row.get("muon_ortho_batch_max", 0) or 0),
                    seq_len=int(args.seq_len),
                    target_tokens_per_update=int(args.target_tokens_per_update),
                    compile_mode=str(compile_mode),
                )
                if _candidate_key(candidate) not in seen:
                    _add_unique_candidate(compile_candidates, compile_seen, candidate)

    for candidate in compile_candidates:
        _run_candidate(
            candidate=candidate,
            args=args,
            total_memory_gb=float(total_memory_gb),
            rows=rows,
            seen=seen,
            phase="compile_refine",
            probe_batch_sizes=probe_batch_sizes,
            refine_batch_sizes=refine_batch_sizes,
            compile_modes=compile_modes,
            baseline_compile_mode=str(baseline_compile_mode),
            compile_refine_top_k=int(compile_refine_top_k),
            auto_batch_limit=int(auto_batch_limit),
            stopped_after_oom=bool(stopped_after_oom),
        )

    valid_rows = [row for row in rows if bool(row.get("ok", False))]
    if not valid_rows:
        raise RuntimeError("gpu sweep produced no valid candidates")
    ranked = sorted(valid_rows, key=_throughput_key, reverse=True)
    best = ranked[0]
    memory_best = max(
        valid_rows,
        key=lambda row: (
            float(row.get("memory_utilization", 0.0) or 0.0),
            float(row.get("tokens_per_s", 0.0) or 0.0),
        ),
    )
    return {
        "kind": "pretrain_gpu_sweep",
        "seq_len": int(args.seq_len),
        "target_tokens_per_update": int(args.target_tokens_per_update),
        "total_memory_gb": float(total_memory_gb),
        "batch_probe_sizes": [int(batch_size) for batch_size in probe_batch_sizes],
        "refine_batch_sizes": [int(batch_size) for batch_size in refine_batch_sizes],
        "compile_modes": [str(mode) for mode in compile_modes],
        "baseline_compile_mode": str(baseline_compile_mode),
        "compile_refine_top_k": int(compile_refine_top_k),
        "auto_batch_limit": int(auto_batch_limit),
        "stopped_after_oom": bool(stopped_after_oom),
        "results": rows,
        "ranked": ranked,
        "best": best,
        "memory_best": memory_best,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep target GPU pretrain batch and loss-chunk candidates."
    )
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/pretrain_tokens/train")
    parser.add_argument("--output_json", type=str, default="/root/autodl-tmp/pretrain_gpu_sweep.json")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Append one JSON row per completed candidate. Empty derives from output_json.",
    )
    parser.add_argument(
        "--partial_json",
        type=str,
        default="",
        help="Atomically updated partial summary. Empty derives from output_json.",
    )
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument(
        "--target_tokens_per_update",
        type=int,
        default=DEFAULT_TARGET_TOKENS_PER_UPDATE,
    )
    parser.add_argument("--batch_sizes", type=str, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--loss_chunks", type=str, default=DEFAULT_LOSS_CHUNKS)
    parser.add_argument(
        "--muon_ortho_batch_maxes",
        type=str,
        default=DEFAULT_MUON_ORTHO_BATCH_MAXES,
    )
    parser.add_argument("--refine_top_k", type=int, default=DEFAULT_REFINE_TOP_K)
    parser.add_argument("--compile_modes", type=str, default=DEFAULT_COMPILE_MODES)
    parser.add_argument(
        "--compile_refine_top_k",
        type=int,
        default=DEFAULT_COMPILE_REFINE_TOP_K,
    )
    parser.add_argument("--auto_expand_batch_sizes", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--max_auto_batch_size",
        type=int,
        default=0,
        help="0 resolves to target_tokens_per_update / seq_len.",
    )
    parser.add_argument(
        "--batch_expand_factor",
        type=float,
        default=DEFAULT_BATCH_EXPAND_FACTOR,
    )
    parser.add_argument("--warmup_updates", type=int, default=1)
    parser.add_argument("--timed_updates", type=int, default=2)
    parser.add_argument("--include_optimizer", type=int, choices=[0, 1], default=1)
    parser.add_argument("--step_execution_backend", type=str, default="inductor")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not str(args.output_jsonl).strip():
        args.output_jsonl = _default_output_jsonl(str(args.output_json))
    if not str(args.partial_json).strip():
        args.partial_json = str(Path(str(args.output_json)).with_suffix(".partial.json"))
    payload = run_sweep(args)
    payload["status"] = "complete"
    out = Path(str(args.output_json))
    _write_json_atomic(out, payload)
    if payload.get("best"):
        print("[BEST] " + json.dumps(payload["best"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
