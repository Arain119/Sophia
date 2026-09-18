#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${SOPHIA_REMOTE_ROOT:-$ROOT}"
PYTHON_BIN="${SOPHIA_PYTHON:-python3.12}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REMOTE_ROOT/ops/reports" "$REMOTE_ROOT/ops/logs"

echo "[1/5] environment"
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "[2/5] torch inductor audit"
"$PYTHON_BIN" -m ml.tooling.scripts.audit_torch_inductor \
  --output_json "$REMOTE_ROOT/ops/reports/torch_inductor_gpu_probe.json"

echo "[3/5] acceleration capabilities"
"$PYTHON_BIN" -m ml.tooling.scripts.audit_acceleration \
  --output_json "$REMOTE_ROOT/ops/reports/acceleration_gpu_probe.json"

echo "[4/5] BF16 precision probe"
"$PYTHON_BIN" -m ml.tooling.scripts.audit_precision_probe \
  --device cuda:0 \
  --seq_len 4096 \
  --output_json "$REMOTE_ROOT/ops/reports/precision_probe_gpu.json"

echo "[5/5] summary"
SOPHIA_PROBE_REPORT_ROOT="$REMOTE_ROOT/ops/reports" "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

root = Path(os.environ["SOPHIA_PROBE_REPORT_ROOT"])
for name in (
    "acceleration_gpu_probe.json",
    "precision_probe_gpu.json",
    "torch_inductor_gpu_probe.json",
):
    path = root / name
    print(f"--- {name}")
    if not path.is_file():
        print("MISSING")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    if name.startswith("precision"):
        for probe in payload.get("probes", []):
            print(probe)
    elif name.startswith("acceleration"):
        print(payload.get("capabilities"))
    else:
        print(payload.get("summary"))
PY
