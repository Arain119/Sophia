#!/usr/bin/env python
"""
Post-hoc checkpoint EMA for Sophia pretrain runs.

Averages the ``model`` state dicts of saved training checkpoints after the run
finishes, at zero training-time cost. The result is written as a weights-only
checkpoint payload (``{"model": ..., "step": <last>}``) that the standard
checkpoint loader and export tooling accept.

Modes:
  - ema:      w_ema <- decay * w_ema + (1 - decay) * w_k, applied in step order.
  - uniform:  plain average of all selected checkpoints ("model soup").

Example:
  PYTHONPATH=. python3 -m ml.tooling.scripts.posthoc_checkpoint_ema \
    --ckpt_dir runs/pretrain_cuda/checkpoints \
    --last 5 --mode ema --decay 0.7 \
    --output runs/pretrain_cuda/posthoc_ema.pt
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

_CKPT_RE = re.compile(r"^(?P<prefix>.*?)(?P<step>\d+)\.pt$")


def discover_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for name in os.listdir(ckpt_dir):
        match = _CKPT_RE.match(name)
        if match is None:
            continue
        out.append((int(match.group("step")), os.path.join(ckpt_dir, name)))
    out.sort(key=lambda item: item[0])
    return out


def load_model_state(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected checkpoint payload type in {path}")
    model_state = payload.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError(f"checkpoint has no `model` state dict: {path}")
    return model_state


def accumulate(
    ema_state: dict[str, torch.Tensor] | None,
    model_state: dict[str, torch.Tensor],
    *,
    mode: str,
    decay: float,
    sample_index: int,
) -> dict[str, torch.Tensor]:
    if ema_state is None:
        return {
            key: value.detach().to(torch.float32).clone()
            if torch.is_floating_point(value)
            else value.detach().clone()
            for key, value in model_state.items()
        }
    if set(ema_state.keys()) != set(model_state.keys()):
        raise RuntimeError("checkpoint state dict keys diverge across checkpoints")
    for key, value in model_state.items():
        target = ema_state[key]
        if not torch.is_floating_point(value):
            ema_state[key] = value.detach().clone()
            continue
        update = value.detach().to(torch.float32)
        if mode == "uniform":
            # Running mean over sample_index+1 states.
            target.mul_(float(sample_index) / float(sample_index + 1)).add_(
                update, alpha=1.0 / float(sample_index + 1)
            )
        else:
            target.mul_(float(decay)).add_(update, alpha=1.0 - float(decay))
    return ema_state


def run(
    *,
    ckpt_dir: str,
    checkpoints: list[str],
    last: int,
    mode: str,
    decay: float,
    output: str,
    out_dtype: str,
) -> None:
    if checkpoints:
        selected: list[tuple[int, str]] = []
        for path in checkpoints:
            match = _CKPT_RE.match(os.path.basename(path))
            step = int(match.group("step")) if match else len(selected)
            selected.append((step, path))
        selected.sort(key=lambda item: item[0])
    else:
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            raise RuntimeError(f"checkpoint directory not found: {ckpt_dir!r}")
        selected = discover_checkpoints(ckpt_dir)
    if int(last) > 0:
        selected = selected[-int(last) :]
    if not selected:
        raise RuntimeError("no checkpoints selected")

    ema_state: dict[str, torch.Tensor] | None = None
    for sample_index, (step, path) in enumerate(selected):
        print(f"[EMA] {sample_index + 1}/{len(selected)} step={step} {path}", flush=True)
        ema_state = accumulate(
            ema_state,
            load_model_state(path),
            mode=str(mode),
            decay=float(decay),
            sample_index=int(sample_index),
        )

    assert ema_state is not None
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[str(out_dtype)]
    for key, value in ema_state.items():
        if torch.is_floating_point(value):
            ema_state[key] = value.to(dtype)

    payload = {
        "model": ema_state,
        "step": int(selected[-1][0]),
        "kind": "posthoc_checkpoint_ema",
        "mode": str(mode),
        "decay": float(decay),
        "source_checkpoints": [path for _, path in selected],
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    tmp = f"{output}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, output)
    print(
        f"[OK] wrote {output} | checkpoints={len(selected)} mode={mode} "
        f"decay={decay} dtype={out_dtype}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        default=[],
        help="Explicit checkpoint .pt path (repeatable; overrides --ckpt_dir discovery).",
    )
    parser.add_argument("--last", type=int, default=0, help="Use only the last N checkpoints (0=all).")
    parser.add_argument("--mode", type=str, default="ema", choices=["ema", "uniform"])
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--out_dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args(argv)
    run(
        ckpt_dir=str(args.ckpt_dir),
        checkpoints=[str(x) for x in args.checkpoints],
        last=int(args.last),
        mode=str(args.mode),
        decay=float(args.decay),
        output=str(args.output),
        out_dtype=str(args.out_dtype),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
