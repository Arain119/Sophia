#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.live_train_dashboard import DEFAULT_RUN_DIR, status


METRIC_STALE_S = 300
CHECKPOINT_STALE_S = 5_400
LOW_GPU_UTIL_PCT = 70
LOW_TOK_S_RATIO = 0.55
CHECKPOINT_ACTIVITY_GRACE_S = 180


def _num(value: object, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return "-"


def _event_message(data: dict) -> str:
    progress = data.get("progress") or {}
    train = data.get("train") or {}
    gpu = data.get("gpu") or {}
    health = data.get("health") or {}
    reasons = health.get("reasons") or []

    step = progress.get("step")
    max_steps = progress.get("max_steps")
    pct = progress.get("pct")
    loss = train.get("loss")
    grad = train.get("grad_norm")
    tok_s = train.get("tok_s")
    gpu_util = gpu.get("util_pct")
    temp = gpu.get("temp_c")
    mem_used = gpu.get("mem_used_mb")
    mem_total = gpu.get("mem_total_mb")
    checkpoint = data.get("latest_checkpoint") or "暂无"
    state = health.get("state") or data.get("state") or "unknown"

    parts = [
        f"训练监视：第 {step}/{max_steps} 步（{_num(pct)}%）",
        f"loss {_num(loss, 4)}",
        f"grad {_num(grad, 3)}",
        f"吞吐 {_num(tok_s, 0)} Token/秒",
        f"GPU {_num(gpu_util, 0)}%",
        f"显存 {_num((mem_used or 0) / 1024, 2)}/{_num((mem_total or 0) / 1024, 2)} GB",
        f"温度 {_num(temp, 0)}°C",
        f"检查点 {checkpoint}",
        f"状态 {state}",
    ]
    if reasons:
        parts.append("告警：" + "；".join(str(x) for x in reasons))
    return "，".join(parts) + "。"


def _last_train_age_s(data: dict) -> float | None:
    train = data.get("train") or {}
    history = data.get("history") or []
    ts = train.get("time")
    if not ts and history:
        ts = history[-1].get("time")
    if isinstance(ts, (int, float)):
        return max(time.time() - float(ts), 0.0)
    return None


def _issues(data: dict) -> list[str]:
    state = data.get("state")
    health = data.get("health") or {}
    progress = data.get("progress") or {}
    gpu = data.get("gpu") or {}
    stats = data.get("history_stats") or {}
    reasons = [str(x) for x in (health.get("reasons") or [])]
    issues: list[str] = []

    if state == "completed":
        return ["训练已完成"]
    if state != "running":
        issues.append(f"训练状态为 {state or 'unknown'}")
    if health.get("state") != "healthy":
        issues.append(f"健康状态为 {health.get('state') or 'unknown'}")
    issues.extend(reasons)

    metric_age = _last_train_age_s(data)
    if metric_age is None:
        start_time = (data.get("start") or {}).get("time")
        in_startup_grace = (
            state == "running"
            and isinstance(start_time, (int, float))
            and max(time.time() - float(start_time), 0.0) <= METRIC_STALE_S
        )
        if not in_startup_grace:
            issues.append("未读取到训练指标时间戳")
    elif metric_age > METRIC_STALE_S:
        issues.append(f"训练指标 {metric_age:.0f}s 未更新")

    ckpt_age = progress.get("checkpoint_age_s")
    if isinstance(ckpt_age, (int, float)) and ckpt_age > CHECKPOINT_STALE_S:
        issues.append(f"checkpoint {ckpt_age / 60:.0f} 分钟未更新")

    util = gpu.get("util_pct")
    checkpoint_active = bool(progress.get("checkpoint_write_active")) or (
        isinstance(ckpt_age, (int, float))
        and ckpt_age <= float(CHECKPOINT_ACTIVITY_GRACE_S)
    )
    if (
        isinstance(util, (int, float))
        and util < LOW_GPU_UTIL_PCT
        and not checkpoint_active
    ):
        issues.append(f"GPU 利用率偏低：{util:.0f}%")

    tok_stats = stats.get("tok_s") or {}
    rolling = tok_stats.get("rolling")
    median = tok_stats.get("median")
    if isinstance(rolling, (int, float)) and isinstance(median, (int, float)) and median > 0:
        if rolling < LOW_TOK_S_RATIO * median:
            issues.append(f"近期吞吐显著下降：{rolling:.0f} vs 中位数 {median:.0f} Token/秒")

    return issues


def _append_event(run_dir: Path, message: str) -> None:
    path = run_dir / "dashboard_events.jsonl"
    row = {"time": time.time(), "message": message}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--launch-log", type=Path)
    parser.add_argument("--interval", type=int, default=1200)
    args = parser.parse_args()

    launch_log = args.launch_log or Path(f"{args.run_dir}.launch.log")
    last_issue_key = ""
    while True:
        data = status(args.run_dir, launch_log)
        issues = _issues(data)
        issue_key = "|".join(issues)
        if issues and issue_key != last_issue_key:
            _append_event(args.run_dir, _event_message(data) + " 触发原因：" + "；".join(issues))
            last_issue_key = issue_key
        elif not issues:
            last_issue_key = ""
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
