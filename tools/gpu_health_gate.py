"""GPU health gate for release pretrain machines.

Verifies DRAM bandwidth and the GEMM shapes this training stack depends on.
Motivation: on 2026-07-02 a release GPU instance silently degraded to ~6 GB/s
DRAM bandwidth while small L2-resident GEMMs still ran at full speed, so smoke
tests passed while real training was ~120x slower.
Run this before starting or resuming any long training run.

Usage:
    python tools/gpu_health_gate.py [--device cuda:0]

Exit code 0 = HEALTHY, 1 = SICK.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

BANDWIDTH_GATE_GB_S = 1200.0
SQUARE_GATE_TFLOPS = 150.0
VOCAB_GATE_TFLOPS = 100.0
MAIN_GATE_TFLOPS = 150.0


def measure_dram_bandwidth_gb_s(device: torch.device) -> float:
    x = torch.randn(1024 * 1024 * 256, device=device)  # 1 GiB fp32
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize(device)
    iters = 30
    start = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    del x, y
    torch.cuda.empty_cache()
    return 2.0 * 1024**3 * 4 * iters / elapsed / 1e9


def measure_gemm_tflops(
    device: torch.device, m: int, k: int, n: int, iters: int
) -> float:
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    del a, b
    torch.cuda.empty_cache()
    return 2.0 * m * k * n * iters / elapsed / 1e12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0")
    parsed = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("VERDICT: SICK (CUDA unavailable)")
        return 1
    device = torch.device(str(parsed.device))
    torch.cuda.set_device(device)

    bandwidth = measure_dram_bandwidth_gb_s(device)
    # Shapes: >L2 square product (Muon Newton-Schulz), vocab projection
    # (lm_head/loss), and the dominant transformer matmul shape.
    square = measure_gemm_tflops(device, 4096, 4096, 4096, iters=200)
    vocab = measure_gemm_tflops(device, 4096, 1536, 49152, iters=50)
    main_shape = measure_gemm_tflops(device, 4096, 1536, 4096, iters=100)

    checks = [
        ("DRAM bandwidth GB/s", bandwidth, BANDWIDTH_GATE_GB_S),
        ("square 4096^3 TFLOPS", square, SQUARE_GATE_TFLOPS),
        ("vocab 4096x1536x49152 TFLOPS", vocab, VOCAB_GATE_TFLOPS),
        ("main 4096x1536x4096 TFLOPS", main_shape, MAIN_GATE_TFLOPS),
    ]
    healthy = True
    for name, value, gate in checks:
        status = "OK " if value >= gate else "BAD"
        healthy = healthy and value >= gate
        print(f"[{status}] {name}: {value:8.1f} (gate >= {gate:g})")
    print("VERDICT:", "HEALTHY" if healthy else "SICK")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
