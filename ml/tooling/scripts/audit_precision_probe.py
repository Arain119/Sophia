from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from ml.modeling import SophiaDecoder
from ml.tasks.pretrain.pipeline import runtime_preflight_config
from ml.tooling.scripts.audit_acceleration import build_acceleration_report
from ml.runtime.torch_env import setup_seed, setup_torch_backends
from ml.training.pretrain.release_config import release_pretrain_runtime_metadata


def _probe_status(
    *,
    mode: str,
    status: str,
    reason: str = "",
    loss: float | None = None,
    grad_norm: float | None = None,
    tokens_per_sec: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": str(mode),
        "status": str(status),
        "reason": str(reason),
    }
    if loss is not None:
        out["loss"] = float(loss)
    if grad_norm is not None:
        out["grad_norm"] = float(grad_norm)
    if tokens_per_sec is not None:
        out["tokens_per_sec"] = float(tokens_per_sec)
    return out


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        value = float(torch.linalg.vector_norm(grad.detach().float()).item())
        total += value * value
    return math.sqrt(total)


def run_bf16_probe(
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> dict[str, Any]:
    if device.type != "cuda":
        return _probe_status(
            mode="bf16",
            status="skipped",
            reason="CUDA is required for the production precision probe.",
        )

    setup_torch_backends()
    setup_seed(int(seed))
    cfg = runtime_preflight_config(batch_size=int(batch_size), seq_len=int(seq_len))
    model = SophiaDecoder(cfg).to(device=device, dtype=torch.bfloat16).train()
    input_ids = torch.randint(
        low=0,
        high=int(cfg.vocab_size),
        size=(int(batch_size), int(seq_len)),
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        out = model(input_ids=input_ids, labels=labels, compute_loss=True)
        loss = out.loss
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = max(float(time.perf_counter() - start), 1e-9)
    loss_value = float(loss.detach().float().item())
    grad_value = float(_grad_norm(model))
    if not (math.isfinite(loss_value) and math.isfinite(grad_value)):
        return _probe_status(
            mode="bf16",
            status="failed",
            reason="non_finite_loss_or_grad",
            loss=loss_value,
            grad_norm=grad_value,
        )
    return _probe_status(
        mode="bf16",
        status="passed",
        reason="finite_forward_backward",
        loss=loss_value,
        grad_norm=grad_value,
        tokens_per_sec=float(int(batch_size) * int(seq_len)) / elapsed,
    )


def build_precision_probe_report(
    *,
    acceleration_report: dict[str, Any],
    bf16_result: dict[str, Any],
) -> dict[str, Any]:
    probes = [dict(bf16_result)]
    bf16_passed = str(bf16_result.get("status")) == "passed"
    return {
        "kind": "precision_probe",
        "contract": release_pretrain_runtime_metadata(),
        "passed": bool(bf16_passed),
        "probes": probes,
        "acceleration": acceleration_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a BF16 forward/backward probe through the pretraining stack."
    )
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seq_len", default=4096, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_json", default="", type=str)
    parsed = parser.parse_args(argv)

    acceleration = build_acceleration_report()
    device = torch.device(str(parsed.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        bf16 = _probe_status(
            mode="bf16",
            status="failed",
            reason="CUDA requested but unavailable.",
        )
    else:
        bf16 = run_bf16_probe(
            device=device,
            batch_size=int(parsed.batch_size),
            seq_len=int(parsed.seq_len),
            seed=int(parsed.seed),
        )
    report = build_precision_probe_report(
        acceleration_report=acceleration,
        bf16_result=bf16,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if str(parsed.output_json or "").strip():
        path = Path(str(parsed.output_json)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
