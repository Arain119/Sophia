#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("/root/autodl-tmp/out/formal_pretrain_rtx5090_bf16")
DEFAULT_TOKENS_PER_UPDATE = 270_336
ERROR_PATTERNS = (
    "Traceback",
    "RuntimeError",
    "Exception",
    "stability guard",
    "Killed",
    "CUDA out",
    "out of memory",
)


def _tail(path: Path, lines: int, max_bytes: int = 2_000_000) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
            handle.readline()
        data = handle.read()
    return data.decode("utf-8", errors="replace").splitlines()[-lines:]


def _load_metrics(
    run_dir: Path, history_max: int = 10_000
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    latest_start: dict[str, Any] = {}
    latest_train: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for line in _tail(run_dir / "metrics.jsonl", 50_000, 24_000_000):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") == "start":
            latest_start = row
        elif row.get("type") == "train":
            latest_train = row
            history.append(row)
    if len(history) > history_max:
        history = history[-history_max:]
    return latest_start, latest_train, history


def _tokens_per_update(start: dict[str, Any]) -> int:
    for key in ("tokens_per_update", "tokens_per_step", "tokens_per_batch"):
        val = start.get(key)
        if val:
            return int(val)
    recipe = start.get("recipe") or {}
    for key in ("tokens_per_update", "tokens_per_step", "tokens_per_batch"):
        val = recipe.get(key)
        if val:
            return int(val)
    return DEFAULT_TOKENS_PER_UPDATE


def _train_pids() -> list[int]:
    pids: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parts = [
                x.decode("utf-8", "ignore")
                for x in (proc / "cmdline").read_bytes().split(b"\0")
                if x
            ]
        except OSError:
            continue
        if any(part == "-m" and i + 1 < len(parts) and parts[i + 1] == "ml.cli.train" for i, part in enumerate(parts)):
            pids.append(int(proc.name))
    return sorted(pids)


def _gpu() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=3).strip()
    except Exception:
        return {}
    if not out:
        return {}
    parts = [x.strip() for x in out.splitlines()[0].split(",")]
    if len(parts) < 6:
        return {}
    return {
        "name": parts[0],
        "mem_used_mb": float(parts[1]),
        "mem_total_mb": float(parts[2]),
        "util_pct": float(parts[3]),
        "temp_c": float(parts[4]),
        "power_w": float(parts[5]),
    }


def _errors(run_dir: Path, launch_log: Path) -> list[str]:
    rows: list[str] = []
    for path in (launch_log, run_dir / "metrics.jsonl"):
        for line in _tail(path, 500):
            if any(pattern in line for pattern in ERROR_PATTERNS):
                rows.append(f"{path.name}: {line}")
    return rows[-20:]


def _checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    rows = []
    for path in ckpt_dir.glob("*.pt"):
        stat = path.stat()
        rows.append({"name": path.name, "gb": stat.st_size / 1024**3, "mtime": stat.st_mtime})
    return sorted(rows, key=lambda row: row["mtime"], reverse=True)[:5]


def _monitor_events(run_dir: Path, limit: int = 10) -> list[str]:
    path = run_dir / "dashboard_events.jsonl"
    rows: list[str] = []
    for line in _tail(path, limit, 200_000):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append(line.strip())
            continue
        text = row.get("message")
        ts = row.get("time")
        if text and ts:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
            rows.append(f"{stamp}｜{text}")
        elif text:
            rows.append(str(text))
    return rows[-limit:]


def _series_stats(values: list[Any], require_positive: bool = False) -> dict[str, Any]:
    vals: list[float] = []
    for v in values:
        if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
            fv = float(v)
            if require_positive and fv <= 0:
                continue
            vals.append(fv)
    if not vals:
        return {"min": None, "max": None, "avg": None, "median": None, "latest": None, "rolling": None, "count": 0}
    rolling = vals[-20:] if len(vals) >= 20 else vals
    return {
        "min": min(vals),
        "max": max(vals),
        "avg": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "latest": vals[-1],
        "rolling": sum(rolling) / len(rolling),
        "count": len(vals),
    }


def _health(
    state: str,
    gpu: dict[str, Any],
    history_stats: dict[str, dict[str, Any]],
    errors: list[str],
    incident: str,
) -> dict[str, Any]:
    health: dict[str, Any] = {"state": "healthy", "reasons": []}
    if state == "needs_attention":
        health["state"] = "attention"
        health["reasons"].append("检测到 incident 或错误日志，需要人工排查")
    elif state == "stopped":
        health["state"] = "unknown"
        health["reasons"].append("未检测到训练进程")
    else:
        temp = gpu.get("temp_c")
        if temp is not None and temp > 85:
            health["state"] = "warning"
            health["reasons"].append(f"GPU 温度 {temp:.1f}°C 偏高")
        util = gpu.get("util_pct")
        tok_latest = history_stats.get("tok_s", {}).get("latest") or 0
        if util is not None and util < 5 and tok_latest > 0:
            health["state"] = "warning"
            health["reasons"].append("GPU 利用率极低但吞吐非零，请检查瓶颈")
        grad = history_stats.get("grad_norm", {}).get("latest")
        if grad is not None and grad > 100:
            health["state"] = "warning"
            health["reasons"].append(f"grad_norm {grad:.2f} 显著偏高")
        loss = history_stats.get("loss", {}).get("latest")
        if loss is not None and not math.isfinite(loss):
            health["state"] = "attention"
            health["reasons"].append("loss 为 Inf/NaN，训练异常")
        median_tok = history_stats.get("tok_s", {}).get("median") or 0
        recent_tok = history_stats.get("tok_s", {}).get("rolling") or 0
        if median_tok and history_stats.get("tok_s", {}).get("count", 0) >= 10 and recent_tok < 0.35 * median_tok:
            health["state"] = "warning"
            health["reasons"].append("近期平均吞吐明显低于历史中位数")
    return health


def status(run_dir: Path, launch_log: Path) -> dict[str, Any]:
    latest_start, latest_train, history = _load_metrics(run_dir)
    pids = _train_pids()
    gpu = _gpu()
    incident_path = run_dir / "stability_incident.json"
    incident = incident_path.read_text(encoding="utf-8", errors="replace") if incident_path.exists() else ""
    errors = _errors(run_dir, launch_log)

    tokens_per_update = _tokens_per_update(latest_start)
    step = int(latest_train.get("step") or 0)
    max_steps = int(latest_train.get("max_steps") or latest_start.get("max_steps") or 0)
    tok_s = float(latest_train.get("tok_s") or 0)
    remain_steps = max(max_steps - step, 0)
    trained_tokens = step * tokens_per_update
    remain_tokens = remain_steps * tokens_per_update
    eta_s = remain_tokens / tok_s if tok_s > 0 else 0

    history_stats = {
        "loss": _series_stats([r.get("loss") for r in history]),
        "tok_s": _series_stats([r.get("tok_s") for r in history], require_positive=True),
        "grad_norm": _series_stats([r.get("grad_norm") for r in history]),
    }
    recent_tok_s = history_stats["tok_s"]["rolling"] or history_stats["tok_s"]["latest"] or tok_s
    eta_s_recent = remain_tokens / recent_tok_s if recent_tok_s > 0 else 0

    if max_steps and step >= max_steps:
        state = "completed"
    elif incident or errors:
        state = "needs_attention"
    elif pids:
        state = "running"
    else:
        state = "stopped"

    checkpoints = _checkpoints(run_dir)
    latest_checkpoint = checkpoints[0]["name"] if checkpoints else ""
    checkpoint_age_s = time.time() - checkpoints[0]["mtime"] if checkpoints else None

    health = _health(state, gpu, history_stats, errors, incident)

    return {
        "time": time.time(),
        "state": state,
        "run_dir": str(run_dir),
        "pids": pids,
        "gpu": gpu,
        "start": latest_start,
        "train": latest_train,
        "history": history,
        "history_stats": history_stats,
        "health": health,
        "progress": {
            "step": step,
            "max_steps": max_steps,
            "pct": step / max_steps * 100 if max_steps else 0,
            "remain_steps": remain_steps,
            "remain_tokens": remain_tokens,
            "trained_tokens": trained_tokens,
            "tokens_per_update": tokens_per_update,
            "eta_s": eta_s,
            "eta_s_recent": eta_s_recent,
            "recent_tok_s": recent_tok_s,
            "checkpoint_age_s": checkpoint_age_s,
        },
        "incident": incident,
        "errors": errors,
        "checkpoints": checkpoints,
        "latest_checkpoint": latest_checkpoint,
        "events": _monitor_events(run_dir) + [
            "旧训练在约第 3870 步出现损失与梯度范数异常，训练稳定性受损。",
            "排查定位：纯 AGC 梯度裁剪与 Muon + AdamW 混合优化器路径不匹配，且恢复训练会覆盖新的裁剪语义。",
            "修复方案：改为 hybrid_auto；Muon 使用全局范数裁剪，AdamW 使用 AGC；恢复训练不再覆盖当前配置。",
            "从 ckpt_step3843 启动探测训练并跑过旧爆点，验证稳定后正式训练已重启并持续运行。",
        ],
    }


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>训练观测台</title>
<style>
:root{
  --bg:#eaddc5;
  --bg-deep:#d8c9ad;
  --paper:#fff9ed;
  --paper-weak:rgba(255,249,237,.72);
  --paper-mid:rgba(255,249,237,.9);
  --ink:#2e291e;
  --ink-light:#5c5340;
  --muted:#8b7d65;
  --rule:rgba(46,41,30,.12);
  --rule-strong:rgba(46,41,30,.24);
  --copper:#a65d2e;
  --copper-light:#c97b4a;
  --copper-wash:rgba(166,93,46,.14);
  --sage:#4f7a5a;
  --sage-light:#6d9c7a;
  --sage-wash:rgba(79,122,90,.14);
  --indigo:#3b4f68;
  --indigo-light:#586f8c;
  --amber:#c9a227;
  --vermilion:#b53c2f;
  --verdigris:#4f8a8a;
  --shadow:0 14px 40px rgba(36,30,20,.18);
  --shadow-soft:0 6px 18px rgba(36,30,20,.10);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;
  background:var(--bg);
  color:var(--ink);
  font:15px/1.65 "Georgia","Times New Roman","Noto Serif CJK SC",serif;
  min-height:100vh;
}
body::before{
  content:"";
  position:fixed;
  inset:0;
  background:
    linear-gradient(rgba(46,41,30,.06) 1px,transparent 1px),
    linear-gradient(90deg,rgba(46,41,30,.06) 1px,transparent 1px),
    radial-gradient(circle at 18% 12%,rgba(166,93,46,.12),transparent 34%),
    radial-gradient(circle at 86% 88%,rgba(79,122,90,.10),transparent 32%),
    linear-gradient(180deg,var(--bg) 0%,var(--bg-deep) 100%);
  background-size:28px 28px,28px 28px,auto,auto,auto;
  filter:url(#paper-grain);
  pointer-events:none;
  z-index:-2;
}
body::after{
  content:"";
  position:fixed;
  inset:0;
  background:
    radial-gradient(circle at 50% 0%,rgba(255,249,237,.35),transparent 55%),
    radial-gradient(circle at 70% 100%,rgba(166,93,46,.06),transparent 40%);
  pointer-events:none;
  z-index:-1;
}
main{max-width:1320px;margin:0 auto;padding:32px 26px;position:relative}

header{position:relative;margin-bottom:20px;padding-bottom:22px;padding-right:170px;border-bottom:1px solid var(--rule);z-index:1}
header .ornament{
  position:absolute;
  right:-22px;
  top:-30px;
  width:170px;
  height:170px;
  opacity:.18;
  pointer-events:none;
  color:var(--copper);
  z-index:0;
}
@media(max-width:820px){header{padding-right:0}header .ornament{width:95px;height:95px;right:2px;top:-6px;opacity:.10}}
h1{
  position:relative;
  z-index:1;
  font-size:34px;
  font-weight:400;
  margin:0 0 6px;
  letter-spacing:.4px;
  color:var(--ink);
  font-variant:small-caps;
}
h1 .subtitle{
  display:block;
  font-size:12px;
  letter-spacing:3px;
  text-transform:uppercase;
  color:var(--copper);
  margin-top:4px;
  font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
#run{
  font-size:12.5px;
  color:var(--muted);
  font-family:"SFMono-Regular",Consolas,monospace;
  word-break:break-all;
  margin-top:8px;
}
.vine-divider{
  display:block;
  width:100%;
  height:28px;
  margin:8px 0 22px;
  color:var(--copper);
  opacity:.22;
  pointer-events:none;
}
.filigrane{
  position:fixed;
  left:-40px;
  bottom:-40px;
  width:460px;
  height:460px;
  color:var(--copper);
  opacity:.05;
  pointer-events:none;
  z-index:-1;
}

.badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:6px 14px;
  border-radius:999px;
  font-size:13px;
  font-weight:600;
  background:var(--paper-weak);
  border:1px solid var(--rule);
  font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  box-shadow:var(--shadow-soft);
}
.dot{width:9px;height:9px;border-radius:50%;box-shadow:0 0 10px currentColor}
.ok{color:var(--sage)}.bad{color:var(--vermilion)}.warn{color:var(--amber)}.unknown{color:var(--muted)}
.ok-bg{background:rgba(79,122,90,.12);border-color:rgba(79,122,90,.38);color:#2a5235}
.bad-bg{background:rgba(181,60,47,.12);border-color:rgba(181,60,47,.38);color:#7c241c}
.warn-bg{background:rgba(201,162,39,.14);border-color:rgba(201,162,39,.40);color:#7a5f0f}
.unknown-bg{background:rgba(139,125,101,.14);border-color:rgba(139,125,101,.35);color:#5c5340}

.status-ribbon{
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  gap:14px;
  margin:18px 0 26px;
  font-size:13px;
  color:var(--muted);
  font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
.status-ribbon .sep{flex:1}

.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:20px 0;align-items:stretch}
.grid-auto{grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.card,.panel{
  background:var(--paper-weak);
  border:1px solid var(--rule);
  border-radius:5px;
  box-shadow:var(--shadow);
  position:relative;
  overflow:hidden;
  backdrop-filter:blur(2px);
}
.card{
  padding:18px 48px 18px 18px;
  transition:border-color .25s ease,transform .2s ease,box-shadow .25s ease;
}
.card::before{
  content:"";
  position:absolute;
  top:0;
  left:0;
  right:0;
  height:2px;
  background:linear-gradient(90deg,var(--copper),var(--sage),var(--indigo));
  opacity:.55;
}
.card:hover{border-color:var(--rule-strong);transform:translateY(-1px);box-shadow:0 18px 46px rgba(36,30,20,.14)}
.card .corner{
  position:absolute;
  right:10px;
  top:10px;
  width:34px;
  height:34px;
  opacity:.16;
  pointer-events:none;
  color:var(--copper);
  z-index:0;
}
.label{
  position:relative;
  z-index:1;
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:1.3px;
  color:var(--copper);
  margin-bottom:10px;
  display:flex;
  align-items:center;
  gap:8px;
  flex-wrap:wrap;
  font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
.label svg{width:14px;height:14px;opacity:.85;flex:0 0 14px}
.value{font-size:26px;font-weight:600;margin-top:2px;line-height:1.2;color:var(--ink);font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.value.small{font-size:18px}
.muted{color:var(--muted);font-size:12.5px;margin-top:8px;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;overflow-wrap:anywhere}
.missing{color:var(--muted);font-style:italic}

.progress-wrap{
  padding:28px;
  margin:22px 0 28px;
  background:
    radial-gradient(circle at 92% 18%,rgba(79,122,90,.13),transparent 30%),
    linear-gradient(180deg,var(--paper-weak),rgba(255,249,237,.58));
  border-left:3px solid var(--sage);
}
.progress-top{
  display:grid;
  grid-template-columns:minmax(0,1fr) auto;
  align-items:end;
  gap:16px;
  margin-bottom:18px;
}
.progress-title{font-size:23px;color:var(--ink);margin:0;font-weight:400;font-variant:small-caps;letter-spacing:.5px}
.progress-sub{font-size:13px;color:var(--muted);font-family:system-ui,sans-serif}
.progress-wrap #pct{
  font-size:42px;
  line-height:1;
  color:var(--sage);
  text-align:right;
  min-width:140px;
}
.progress-bar{
  height:18px;
  background:rgba(46,41,30,.10);
  border:1px solid var(--rule);
  border-radius:999px;
  overflow:hidden;
  position:relative;
  box-shadow:inset 0 1px 2px rgba(46,41,30,.12);
}
.progress-bar .fill{
  height:100%;
  border-radius:999px;
  background:linear-gradient(90deg,var(--copper),var(--sage),var(--verdigris));
  box-shadow:inset 0 0 12px rgba(255,249,237,.22);
  width:0;
  transition:width .9s cubic-bezier(.22,1,.36,1);
  position:relative;
  overflow:hidden;
}
.progress-bar .fill::after{
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,249,237,.28),transparent);
  animation:shimmer 2.5s infinite linear;
}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.progress-stats{
  display:grid;
  grid-template-columns:repeat(5,minmax(0,1fr));
  gap:0;
  margin-top:18px;
  border-top:1px dashed rgba(166,93,46,.28);
}
.progress-stat{
  min-width:0;
  padding:14px 16px 0;
  border-right:1px solid rgba(46,41,30,.08);
}
.progress-stat:first-child{padding-left:0}
.progress-stat:last-child{border-right:none;padding-right:0}
.progress-stat .label{
  font-size:10px;
  margin-bottom:6px;
  color:var(--muted);
  letter-spacing:1px;
}
.progress-stat .value.small{
  font-size:20px;
  color:var(--ink);
}
.progress-stat .muted{
  font-size:11.5px;
  margin-top:4px;
}

.panel{padding:22px;margin-bottom:20px}
.panel h3{
  font-size:17px;
  font-weight:400;
  margin:0 0 16px;
  color:var(--ink);
  display:flex;
  align-items:center;
  gap:10px;
  font-variant:small-caps;
  letter-spacing:1px;
}
.panel h3 svg{width:18px;height:18px;opacity:.85}
.panel h3::after{
  content:"";
  flex:1;
  height:1px;
  background:linear-gradient(90deg,var(--rule),transparent);
}

.chart-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:24px;margin:28px 0;align-items:stretch}
.chart-card{
  background:var(--paper-mid);
  border:1px solid var(--rule);
  border-radius:5px;
  padding:18px;
  box-shadow:var(--shadow);
  position:relative;
  overflow:hidden;
  display:flex;
  flex-direction:column;
  min-height:420px;
}
.chart-card::before{
  content:"";
  position:absolute;
  inset:12px;
  pointer-events:none;
  border:0;
  border-radius:8px;
  z-index:0;
}
.chart-card .plate-watermark{
  position:absolute;
  right:14px;
  top:14px;
  width:84px;
  height:42px;
  color:var(--copper);
  opacity:.06;
  pointer-events:none;
  z-index:0;
}
.chart-head{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:14px;
  margin:0 0 12px;
  position:relative;
  z-index:1;
  flex:0 0 auto;
}
.chart-card h4{
  margin:0;
  font-size:12px;
  text-transform:uppercase;
  letter-spacing:1.4px;
  color:var(--copper);
  font-weight:600;
  font-family:system-ui,sans-serif;
  white-space:nowrap;
}
.chart-summary{
  flex:1 1 auto;
  margin-left:auto;
  color:var(--muted);
  font:12px/1.35 system-ui,sans-serif;
  text-align:right;
  min-width:0;
  max-width:64%;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.chart-controls{
  display:flex;
  align-items:center;
  gap:10px;
  flex:1 1 auto;
  justify-content:flex-end;
  min-width:0;
}
.chart-controls .chart-summary{
  max-width:none;
}
.chart-toggle{
  display:inline-flex;
  align-items:center;
  gap:4px;
  padding:3px;
  border:1px solid rgba(92,83,64,.20);
  border-radius:999px;
  background:rgba(255,249,237,.62);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.45);
  flex:0 0 auto;
}
.chart-toggle button{
  appearance:none;
  border:0;
  border-radius:999px;
  background:transparent;
  color:var(--muted);
  cursor:pointer;
  font:12px/1 system-ui,sans-serif;
  padding:7px 10px;
  transition:background .18s ease,color .18s ease,box-shadow .18s ease;
}
.chart-toggle button.active{
  background:var(--ink);
  color:var(--paper);
  box-shadow:0 2px 8px rgba(46,41,30,.18);
}
.chart-body{
  position:relative;
  flex:1 1 auto;
  min-height:270px;
  width:100%;
  border-radius:8px;
  background:
    linear-gradient(rgba(46,41,30,.05) 1px,transparent 1px),
    linear-gradient(90deg,rgba(46,41,30,.05) 1px,transparent 1px),
    rgba(255,249,237,.65);
  background-size:24px 24px,24px 24px,auto;
  border:1px solid rgba(46,41,30,.08);
  overflow:hidden;
}
.chart-canvas{
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  display:block;
  z-index:1;
}
.chart-tooltip{
  position:fixed;
  z-index:20;
  display:none;
  pointer-events:none;
  min-width:150px;
  padding:9px 11px;
  background:rgba(255,249,237,.96);
  border:1px solid rgba(166,93,46,.34);
  border-radius:6px;
  box-shadow:0 10px 24px rgba(36,30,20,.18);
  color:var(--ink);
  font:12px/1.45 system-ui,sans-serif;
}
.chart-tooltip b{
  display:block;
  margin-bottom:3px;
  color:var(--copper);
  font-size:11px;
  letter-spacing:.6px;
}

.observatory{
  position:absolute;
  inset:0;
  display:grid;
  grid-template-columns:1.18fr .82fr;
  grid-template-rows:1fr auto;
  gap:18px;
  width:100%;
  min-height:100%;
  padding:18px;
}
.gauge-cluster{
  position:relative;
  min-height:236px;
  display:grid;
  place-items:center;
  border-right:1px dashed rgba(166,93,46,.24);
  padding-right:16px;
  --dial-d:min(94%,244px);
  --primary-d:50%;
  --secondary-d:29%;
}
.dial-field{
  position:relative;
  width:var(--dial-d);
  aspect-ratio:1;
}
.dial-field::before{
  content:"";
  position:absolute;
  left:50%;
  top:50%;
  transform:translate(-50%,-50%);
  width:100%;
  aspect-ratio:1;
  border:1px solid rgba(166,93,46,.18);
  border-radius:50%;
  background:
    radial-gradient(circle,transparent 58%,rgba(166,93,46,.07) 59%,transparent 61%),
    repeating-conic-gradient(from -12deg,rgba(46,41,30,.08) 0 1deg,transparent 1deg 12deg);
  pointer-events:none;
}
.mini-gauge{
  text-align:center;
  min-width:0;
  position:absolute;
  z-index:2;
  width:var(--secondary-d);
  aspect-ratio:1;
  display:grid;
  place-items:center;
}
.mini-gauge.primary-gauge{
  left:50%;
  top:50%;
  transform:translate(-50%,-50%);
  width:var(--primary-d);
}
.mini-gauge.secondary-a{
  left:24%;
  top:76%;
  transform:translate(-50%,-50%);
}
.mini-gauge.secondary-b{
  left:76%;
  top:24%;
  transform:translate(-50%,-50%);
}
.mini-gauge svg{width:100%;height:100%;margin:0 auto;display:block}
.mini-gauge .gauge-label{
  fill:var(--muted);
  font:600 8px system-ui,sans-serif;
  letter-spacing:.4px;
}
.mini-gauge .gauge-value{
  fill:var(--ink);
  font:700 18px system-ui,sans-serif;
  font-variant-numeric:tabular-nums;
}
.observatory-metrics{
  display:grid;
  align-content:center;
  gap:10px;
  min-width:0;
}
.obs-row{
  display:flex;
  justify-content:space-between;
  gap:12px;
  padding:10px 0;
  border-bottom:1px solid rgba(46,41,30,.08);
  font:12px system-ui,sans-serif;
  color:var(--muted);
}
.obs-row b{font-size:13px;color:var(--ink);font-weight:600;text-align:right;overflow-wrap:anywhere;min-width:0}
.observatory-summary{
  grid-column:1/-1;
  width:100%;
  align-self:end;
  display:flex;
  justify-content:space-between;
  gap:14px;
  text-align:left;
  color:var(--muted);
  font-size:13px;
  padding-top:12px;
  border-top:1px dashed rgba(166,93,46,.25);
  overflow-wrap:anywhere;
}
.observatory-summary b{color:var(--ink);font-weight:600}

.expedition-log{
  background:linear-gradient(180deg,rgba(255,249,237,.78),rgba(255,249,237,.55));
  border:1px solid var(--rule);
  border-left:3px solid var(--copper);
}
.expedition-log ol{
  margin:0;
  padding-left:22px;
  color:var(--ink-light);
}
.expedition-log li{
  margin:12px 0;
  padding-left:8px;
  line-height:1.7;
}
.expedition-log li::marker{color:var(--copper);font-weight:700}

table{width:100%;border-collapse:separate;border-spacing:0;font-size:13.5px}
th,td{padding:12px 10px;text-align:left;border-bottom:1px solid var(--rule)}
th{color:var(--copper);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.8px;font-family:system-ui,sans-serif}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(166,93,46,.05)}
pre{
  white-space:pre-wrap;
  background:rgba(46,41,30,.06);
  padding:16px;
  border-radius:4px;
  max-height:300px;
  overflow:auto;
  font:12.5px/1.6 "SFMono-Regular",Consolas,monospace;
  color:var(--ink-light);
  border:1px solid var(--rule);
}
.empty{color:var(--muted);font-style:italic;text-align:center;padding:18px 0}
@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.4}}
.pulse{animation:pulse-dot 2s infinite}

@media(max-width:1100px){
  .grid{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:760px){
  main{padding:18px 14px}
  h1{font-size:25px}
  .grid{grid-template-columns:1fr;gap:14px}
  .chart-grid{grid-template-columns:1fr;gap:18px}
  .chart-card{min-height:340px;padding:16px}
  .chart-body{min-height:230px}
  .chart-head{padding-right:70px}
  .card{padding:16px 44px 16px 16px}
  .progress-wrap{padding:22px}
  .progress-top{grid-template-columns:1fr;gap:10px}
  .progress-wrap #pct{text-align:left;font-size:34px}
  .progress-stats{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 0}
  .progress-stat:nth-child(2n){border-right:none;padding-right:0}
  .progress-stat:nth-child(2n+1){padding-left:0}
  .value{font-size:22px}
  .observatory{position:relative;grid-template-columns:1fr;grid-template-rows:auto auto auto;padding:14px;gap:12px}
  .gauge-cluster{border-right:none;border-bottom:1px dashed rgba(166,93,46,.24);padding:0 0 14px;min-height:212px;--dial-d:min(94%,216px);--primary-d:50%;--secondary-d:29%}
  .mini-gauge .gv{font-size:15px}
}
@media(max-width:420px){
  body{font-size:14px}
  .value{font-size:20px}
  .chart-body{min-height:210px}
  .observatory{gap:10px}
  .gauge-cluster{min-height:196px;--dial-d:min(94%,198px);--primary-d:50%;--secondary-d:29%}
  .observatory-summary{flex-direction:column;gap:4px}
  .chart-head{flex-direction:column;align-items:stretch;gap:5px;padding-right:0}
  .chart-summary{max-width:100%;text-align:right}
}
</style>
</head>
<body>
<svg width="0" height="0" aria-hidden="true" style="position:absolute">
  <defs>
    <filter id="paper-grain" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.7" numOctaves="3" stitchTiles="stitch" result="noise"/>
      <feColorMatrix type="matrix" values="0 0 0 0 0.95 0 0 0 0 0.92 0 0 0 0 0.85 0 0 0 0.10 0" in="noise" result="grain"/>
      <feBlend mode="multiply" in="grain" in2="SourceGraphic"/>
    </filter>
    <filter id="ink-rough">
      <feTurbulence type="fractalNoise" baseFrequency="0.04 0.4" numOctaves="2" result="noise"/>
      <feDisplacementMap in="SourceGraphic" in2="noise" scale="1.6" xChannelSelector="R" yChannelSelector="G"/>
    </filter>
  </defs>
</svg>

<main>
<header>
  <svg class="ornament" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" filter="url(#ink-rough)">
    <g fill="none" stroke="currentColor" stroke-width=".7">
      <circle cx="100" cy="100" r="78"/>
      <circle cx="100" cy="100" r="64"/>
      <circle cx="100" cy="100" r="50"/>
      <circle cx="100" cy="100" r="36"/>
    </g>
    <g stroke="currentColor" stroke-width=".5" opacity=".7">
      <path d="M100 12v22M100 166v22M12 100h22M166 100h22"/>
      <path d="M35 35l16 16M149 149l16 16M35 165l16-16M149 51l16-16"/>
    </g>
    <g fill="currentColor" opacity=".7">
      <circle cx="100" cy="54" r="2.2"/><circle cx="100" cy="146" r="2.2"/>
      <circle cx="54" cy="100" r="2.2"/><circle cx="146" cy="100" r="2.2"/>
    </g>
    <path d="M100 100c-10-18-7-36 10-46 5-3 13-4 21-3" fill="none" stroke="currentColor" stroke-width=".9" opacity=".6"/>
    <path d="M100 100c18 10 36 7 46-10 3-5 4-13 3-21" fill="none" stroke="currentColor" stroke-width=".9" opacity=".6"/>
    <path d="M100 100l48-18" stroke="currentColor" stroke-width=".8" opacity=".5"/>
    <circle cx="100" cy="100" r="4" fill="currentColor" opacity=".5"/>
  </svg>
  <h1>训练观测台 <span class="subtitle">模型训练实时观测</span></h1>
  <div id="run"></div>
</header>

<svg class="vine-divider" viewBox="0 0 1200 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" filter="url(#ink-rough)">
  <path d="M0 20c60-8 120-18 180-10s120 24 180 18 120-28 180-22 120 22 180 16 120-20 180-14 120 18 180 12 120-16 180-10" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <path d="M60 20c10-12 22-12 30 0s18 14 26 0" fill="none" stroke="currentColor" stroke-width=".8"/>
  <path d="M360 22c8-14 20-14 28 0s16 16 24 0" fill="none" stroke="currentColor" stroke-width=".8"/>
  <path d="M660 20c10-12 22-12 30 0s18 14 26 0" fill="none" stroke="currentColor" stroke-width=".8"/>
  <path d="M960 22c8-14 20-14 28 0s16 16 24 0" fill="none" stroke="currentColor" stroke-width=".8"/>
</svg>

<svg class="filigrane" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" filter="url(#ink-rough)">
  <g fill="none" stroke="currentColor" stroke-width=".6">
    <path d="M20 320c60-30 120-20 180-50s100-60 160-40"/>
    <path d="M10 360c70-20 140-40 210-20s120 10 170-30"/>
    <path d="M40 280c50-10 100 10 150-20s90-40 140-20"/>
    <path d="M300 380c20-40 40-90 20-140s-30-100 0-140"/>
  </g>
</svg>

<div class="status-ribbon">
  <span id="stateBadge" class="badge"><span class="dot"></span><span>-</span></span>
  <span id="healthBadge" class="badge unknown-bg"><span class="dot unknown"></span><span>健康状态未知</span></span>
  <span id="pidLine">PID: <b class="missing">未检测到</b></span>
  <span class="sep"></span>
  <span>面板更新: <span id="updateTime">-</span></span>
</div>

<section class="panel progress-wrap">
  <div class="progress-top">
    <div><h2 class="progress-title">训练进度</h2><div class="progress-sub" id="progressSub">暂无训练进度数据</div></div>
    <div class="value" id="pct">-</div>
  </div>
  <div class="progress-bar"><div class="fill" id="fill"></div></div>
  <div class="progress-stats">
    <div class="progress-stat">
      <div class="label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12h18M3 6h18M3 18h18"/></svg>当前步数</div>
      <div class="value small" id="step">-</div>
    </div>
    <div class="progress-stat">
      <div class="label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>剩余步数</div>
      <div class="value small" id="remainSteps">-</div>
    </div>
    <div class="progress-stat">
      <div class="label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>已训练 token</div>
      <div class="value small" id="trainedTokens">-</div>
    </div>
    <div class="progress-stat">
      <div class="label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>剩余 token</div>
      <div class="value small" id="remainTokens">-</div>
    </div>
    <div class="progress-stat">
      <div class="label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>预计完成</div>
      <div class="value small" id="eta">-</div>
      <div class="muted" id="etaDetail"></div>
    </div>
  </div>
</section>

<section class="chart-grid">
  <div class="chart-card">
    <svg class="plate-watermark" viewBox="0 0 120 60" aria-hidden="true"><path d="M0 45c20-10 40-25 60-15s40 20 60 5" fill="none" stroke="currentColor" stroke-width=".7"/><path d="M10 55c20-8 40-18 60-10s35 12 50 2" fill="none" stroke="currentColor" stroke-width=".6"/></svg>
    <div class="chart-head"><h4>损失趋势</h4><span class="chart-summary" id="lossChartNote">暂无历史数据</span></div>
    <div class="chart-body"><canvas id="lossChart" class="chart-canvas"></canvas></div>
  </div>
  <div class="chart-card">
    <svg class="plate-watermark" viewBox="0 0 120 60" aria-hidden="true"><path d="M0 45c20-10 40-25 60-15s40 20 60 5" fill="none" stroke="currentColor" stroke-width=".7"/><path d="M10 55c20-8 40-18 60-10s35 12 50 2" fill="none" stroke="currentColor" stroke-width=".6"/></svg>
    <div class="chart-head"><h4>吞吐趋势</h4><span class="chart-summary" id="toksChartNote">暂无历史数据</span></div>
    <div class="chart-body"><canvas id="toksChart" class="chart-canvas"></canvas></div>
  </div>
  <div class="chart-card">
    <svg class="plate-watermark" viewBox="0 0 120 60" aria-hidden="true"><path d="M0 45c20-10 40-25 60-15s40 20 60 5" fill="none" stroke="currentColor" stroke-width=".7"/><path d="M10 55c20-8 40-18 60-10s35 12 50 2" fill="none" stroke="currentColor" stroke-width=".6"/></svg>
    <div class="chart-head">
      <h4>梯度范数趋势</h4>
      <div class="chart-controls">
        <span class="chart-summary" id="gradChartNote">暂无历史数据</span>
        <div class="chart-toggle" aria-label="梯度图缩放">
          <button type="button" data-grad-scale="body" class="active">主体</button>
          <button type="button" data-grad-scale="full">完整</button>
        </div>
      </div>
    </div>
    <div class="chart-body"><canvas id="gradChart" class="chart-canvas"></canvas></div>
  </div>
  <div class="chart-card">
    <svg class="plate-watermark" viewBox="0 0 120 60" aria-hidden="true"><path d="M0 45c20-10 40-25 60-15s40 20 60 5" fill="none" stroke="currentColor" stroke-width=".7"/><path d="M10 55c20-8 40-18 60-10s35 12 50 2" fill="none" stroke="currentColor" stroke-width=".6"/></svg>
    <div class="chart-head"><h4>GPU 状态</h4></div>
    <div class="chart-body">
      <div class="observatory">
        <div class="gauge-cluster">
          <div class="dial-field">
          <div class="mini-gauge primary-gauge">
            <svg viewBox="0 0 100 100" filter="url(#ink-rough)">
              <circle cx="50" cy="50" r="44" fill="none" stroke="rgba(46,41,30,.12)" stroke-width="7"/>
              <circle id="gUtilArc" cx="50" cy="50" r="44" fill="none" stroke="var(--sage)" stroke-width="7" stroke-linecap="round" stroke-dasharray="276.46" stroke-dashoffset="276.46" transform="rotate(-90 50 50)"/>
              <text class="gauge-label" x="50" y="41" text-anchor="middle">GPU 利用率</text>
              <text id="gUtilText" class="gauge-value" x="50" y="62" text-anchor="middle">-</text>
            </svg>
          </div>
          <div class="mini-gauge secondary-a">
            <svg viewBox="0 0 100 100" filter="url(#ink-rough)">
              <circle cx="50" cy="50" r="44" fill="none" stroke="rgba(46,41,30,.12)" stroke-width="7"/>
              <circle id="gVramArc" cx="50" cy="50" r="44" fill="none" stroke="var(--copper)" stroke-width="7" stroke-linecap="round" stroke-dasharray="276.46" stroke-dashoffset="276.46" transform="rotate(-90 50 50)"/>
              <text class="gauge-label" x="50" y="41" text-anchor="middle">显存</text>
              <text id="gVramText" class="gauge-value" x="50" y="62" text-anchor="middle">-</text>
            </svg>
          </div>
          <div class="mini-gauge secondary-b">
            <svg viewBox="0 0 100 100" filter="url(#ink-rough)">
              <circle cx="50" cy="50" r="44" fill="none" stroke="rgba(46,41,30,.12)" stroke-width="7"/>
              <circle id="gTempArc" cx="50" cy="50" r="44" fill="none" stroke="var(--vermilion)" stroke-width="7" stroke-linecap="round" stroke-dasharray="276.46" stroke-dashoffset="276.46" transform="rotate(-90 50 50)"/>
              <text class="gauge-label" x="50" y="41" text-anchor="middle">温度</text>
              <text id="gTempText" class="gauge-value" x="50" y="62" text-anchor="middle">-</text>
            </svg>
          </div>
          </div>
        </div>
        <div class="observatory-metrics">
          <div class="obs-row"><span>设备</span><b id="gName">未检测到</b></div>
          <div class="obs-row"><span>功耗</span><b id="gPower">-</b></div>
          <div class="obs-row"><span>显存占用</span><b id="gVramDetail">-</b></div>
          <div class="obs-row"><span>温度 / 功耗</span><b id="gTempDetail">-</b></div>
          <div class="obs-row"><span>训练吞吐</span><b id="gTokDetail">-</b></div>
        </div>
        <div class="observatory-summary"><span><b id="gObsNote">等待 GPU 采样</b></span><span>更新时间：<b id="gObsTime">-</b></span></div>
      </div>
    </div>
  </div>
</section>

<section class="panel expedition-log">
  <h3><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>观测记录</h3>
  <ol id="events"></ol>
</section>

<section class="panel">
  <h3><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>检查点归档</h3>
  <table>
    <thead><tr><th>名称</th><th>大小</th><th>时间</th></tr></thead>
    <tbody id="ckpts"></tbody>
  </table>
</section>

</main>
<div id="chartTooltip" class="chart-tooltip"></div>
<script>
const $=id=>document.getElementById(id);
const nf=new Intl.NumberFormat('zh-CN',{maximumFractionDigits:2});
const nfi=new Intl.NumberFormat('zh-CN',{maximumFractionDigits:0});
function present(v,fallback='暂无/未检测到'){return (v===null||v===undefined||v===''||Number.isNaN(v))?fallback:v}
function dur(s){
  if(!s||s<=0)return'暂无';
  let d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);
  if(d>0)return `${d}天${h}小时${m}分`;
  if(h>0)return `${h}小时${m}分`;
  return `${m}分`;
}
function age(s){
  if(s===null||s===undefined||Number.isNaN(s))return'未知';
  if(s<60)return `${Math.floor(s)}秒前`;
  if(s<3600)return `${Math.floor(s/60)}分钟前`;
  if(s<86400)return `${Math.floor(s/3600)}小时前`;
  return `${Math.floor(s/86400)}天前`;
}
function dt(t){return t?new Date(t*1000).toLocaleString('zh-CN',{hour12:false}):'-'}
function stateMeta(state){
  if(state==='running') return {text:'运行中',cls:'ok',dot:'ok'};
  if(state==='completed') return {text:'训练完成',cls:'ok',dot:'ok'};
  if(state==='needs_attention') return {text:'需要处理',cls:'bad',dot:'bad'};
  return {text:'已停止',cls:'warn',dot:'warn'};
}
function healthMeta(health){
  const s=(health&&health.state)||'unknown';
  if(s==='healthy') return {text:'训练健康',cls:'ok',dot:'ok'};
  if(s==='attention') return {text:'需要关注',cls:'bad',dot:'bad'};
  if(s==='warning') return {text:'存在警告',cls:'warn',dot:'warn'};
  return {text:'状态未知',cls:'unknown',dot:'unknown'};
}
function fmtTokens(n){
  if(n===null||n===undefined||Number.isNaN(n))return'暂无';
  if(n>=1e12)return nf.format(n/1e12)+' T';
  if(n>=1e9)return nf.format(n/1e9)+' B';
  if(n>=1e6)return nf.format(n/1e6)+' M';
  if(n>=1e3)return nf.format(n/1e3)+' K';
  return nf.format(n);
}

function resizeCanvas(canvas){
  const wrap=canvas.parentElement;
  const rect=(wrap&&wrap.getBoundingClientRect?wrap:canvas).getBoundingClientRect();
  const dpr=window.devicePixelRatio||1;
  const cssW=Math.max(1,Math.floor(rect.width||canvas.clientWidth||320));
  const cssH=Math.max(1,Math.floor(rect.height||canvas.clientHeight||210));
  const targetW=Math.max(1,Math.floor(cssW*dpr));
  const targetH=Math.max(1,Math.floor(cssH*dpr));
  if(canvas.width!==targetW) canvas.width=targetW;
  if(canvas.height!==targetH) canvas.height=targetH;
  const ctx=canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,w:cssW,h:cssH};
}

function movingAvg(series,w){
  if(!series||!series.length)return[];
  const out=[]; let sum=0;
  for(let i=0;i<series.length;i++){
    sum+=series[i].y;
    if(i>=w) sum-=series[i-w].y;
    out.push({x:series[i].x,y:sum/Math.min(w,i+1)});
  }
  return out;
}

function catmullRom2bezier(points){
  if(points.length<2)return[];
  const segs=[];
  for(let i=0;i<points.length-1;i++){
    const p0=points[i===0?0:i-1];
    const p1=points[i];
    const p2=points[i+1];
    const p3=points[i+2]||p2;
    const cp1x=p1.x+(p2.x-p0.x)/6;
    const cp1y=p1.y+(p2.y-p0.y)/6;
    const cp2x=p2.x-(p3.x-p1.x)/6;
    const cp2y=p2.y-(p3.y-p1.y)/6;
    segs.push([p1,{x:cp1x,y:cp1y},{x:cp2x,y:cp2y},p2]);
  }
  return segs;
}

function fmtChartNumber(v,dec){
  if(!Number.isFinite(v))return'-';
  if(Math.abs(v)>=1000)return nf.format(v);
  return Number(v).toFixed(dec);
}

function quantile(sortedVals,q){
  if(!sortedVals.length)return NaN;
  const pos=(sortedVals.length-1)*q;
  const lo=Math.floor(pos),hi=Math.ceil(pos);
  if(lo===hi)return sortedVals[lo];
  return sortedVals[lo]+(sortedVals[hi]-sortedVals[lo])*(pos-lo);
}

function filterRobustOutliers(series,q=.98){
  if(!series||series.length<20)return series||[];
  const vals=series.map(d=>d.y).filter(Number.isFinite).sort((a,b)=>a-b);
  if(vals.length<20)return series;
  const qv=quantile(vals,q);
  const med=quantile(vals,.5);
  const max=vals[vals.length-1];
  if(!Number.isFinite(qv)||!Number.isFinite(med)||!(max>qv*1.6))return series;
  const upper=Math.max(qv,med*1.35);
  const filtered=series.filter(d=>d.y<=upper);
  return filtered.length>=2?filtered:series;
}

function hideChartTooltip(){
  const tip=$('chartTooltip');
  if(tip) tip.style.display='none';
}

function bindChartTooltip(canvas){
  if(canvas.__tooltipBound)return;
  canvas.__tooltipBound=true;
  canvas.addEventListener('mousemove',event=>{
    const meta=canvas.__chartMeta;
    const tip=$('chartTooltip');
    if(!meta||!tip||!meta.points||!meta.points.length){hideChartTooltip();return;}
    const rect=canvas.getBoundingClientRect();
    const mx=event.clientX-rect.left;
    const my=event.clientY-rect.top;
    const plot=meta.plot;
    if(mx<plot.left||mx>plot.right||my<plot.top||my>plot.bottom){hideChartTooltip();return;}
    let nearest=null,best=Infinity;
    for(const p of meta.points){
      const dx=p.x-mx,dy=p.y-my;
      const score=dx*dx+dy*dy*.35;
      if(score<best){best=score;nearest=p;}
    }
    if(!nearest||Math.abs(nearest.x-mx)>18){hideChartTooltip();return;}
    const unit=meta.unit?` ${meta.unit}`:'';
    const rows=[
      `<b>${meta.label}</b>`,
      `<div>步数：${nf.format(nearest.step)}</div>`,
      `<div>数值：${fmtChartNumber(nearest.value,meta.decimals)}${unit}</div>`,
    ];
    if(nearest.time)rows.push(`<div>时间：${dt(nearest.time)}</div>`);
    if(Number.isFinite(nearest.lr))rows.push(`<div>学习率：${nearest.lr.toExponential(2)}</div>`);
    if(Number.isFinite(nearest.mem_gb))rows.push(`<div>训练显存：${nf.format(nearest.mem_gb)} GB</div>`);
    tip.innerHTML=rows.join('');
    tip.style.display='block';
    const margin=14;
    const tw=tip.offsetWidth||160;
    const th=tip.offsetHeight||90;
    let left=event.clientX+margin;
    let top=event.clientY+margin;
    if(left+tw>window.innerWidth-8)left=event.clientX-tw-margin;
    if(top+th>window.innerHeight-8)top=event.clientY-th-margin;
    tip.style.left=Math.max(8,left)+'px';
    tip.style.top=Math.max(8,top)+'px';
  });
  canvas.addEventListener('mouseleave',hideChartTooltip);
}

function drawSmoothChart(canvas,series,options={}){
  try{
    const {ctx,w,h}=resizeCanvas(canvas);
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle='rgba(255,249,237,.65)';
    ctx.fillRect(0,0,w,h);
    if(!series||series.length<2){
      ctx.fillStyle='#8b7d65';
      ctx.font='13px system-ui,sans-serif';
      ctx.textAlign='center';
      ctx.fillText('暂无足够历史数据',w/2,h/2);
      return;
    }
    const pad={top:28,right:42,bottom:48,left:72};
    const gw=w-pad.left-pad.right;
    const gh=h-pad.top-pad.bottom;
    const vals=series.map(d=>d.y);
    let min=Math.min(...vals),max=Math.max(...vals);
    const isLog=options.yScale==='log';
    let displayMax=max;
    if(options.robustY&&!isLog&&vals.length>=20){
      const sorted=[...vals].sort((a,b)=>a-b);
      const q=quantile(sorted,options.robustQuantile||0.98);
      const med=quantile(sorted,0.5);
      if(Number.isFinite(q)&&Number.isFinite(med)&&max>q*1.6){
        displayMax=Math.max(q,med*1.35);
      }
    }
    if(options.zeroBaseline&&!isLog){min=0;}
    if(min===displayMax){
      if(displayMax===0){displayMax=1;}
      else if(options.zeroBaseline&&!isLog){displayMax*=1.08;}
      else{min*=0.95;displayMax*=1.05;}
    }
    if(isLog){min=Math.max(min,1e-9);}
    const margin=(displayMax-min)*0.08;
    let lo=(options.zeroBaseline&&!isLog)?0:min-margin,hi=displayMax+margin;
    if(isLog){lo=Math.max(lo,1e-9);}
    const range=hi-lo||1;
    const x0=options.xStart===undefined?series[0].x:options.xStart;
    const x1=options.xEnd===undefined?series[series.length-1].x:options.xEnd;
    const xRange=(x1-x0)||1;
    const xAtValue=x=>pad.left+gw*((x-x0)/xRange);
    const xAt=i=>xAtValue(series[i].x);
    const yAt=v=>{
      const shown=options.robustY&&!isLog?Math.min(v,hi):v;
      if(isLog){
        const t=(Math.log10(Math.max(shown,lo))-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo));
        return pad.top+gh*(1-Math.max(0,Math.min(1,t)));
      }
      return pad.top+gh*(1-(shown-lo)/range);
    };

    // fine coordinate grid
    ctx.save();
    ctx.strokeStyle='rgba(46,41,30,.10)';
    ctx.lineWidth=1;
    ctx.setLineDash([2,4]);
    for(let i=0;i<=5;i++){
      const y=pad.top+gh*(i/5);
      ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(w-pad.right,y);ctx.stroke();
    }
    for(let i=0;i<=6;i++){
      const x=pad.left+gw*(i/6);
      ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,h-pad.bottom);ctx.stroke();
    }
    ctx.restore();

    // atlas axes
    ctx.strokeStyle='rgba(46,41,30,.35)';
    ctx.lineWidth=1.2;
    ctx.beginPath();ctx.moveTo(pad.left,pad.top);ctx.lineTo(pad.left,h-pad.bottom);ctx.lineTo(w-pad.right,h-pad.bottom);ctx.stroke();

    // y-axis labels
    ctx.fillStyle='#8b7d65';
    ctx.font='10px system-ui,sans-serif';
    ctx.textAlign='right';
    ctx.textBaseline='middle';
    const dec=options.decimals===undefined?4:options.decimals;
    for(let i=0;i<=5;i++){
      const v=isLog?lo*Math.pow(hi/lo,1-i/5):lo+range*(1-i/5);
      const label=options.yLabelFormatter?options.yLabelFormatter(v):v.toFixed(dec);
      ctx.fillText(label,pad.left-12,pad.top+gh*(i/5));
    }
    // x-axis labels
    ctx.textAlign='left';ctx.textBaseline='alphabetic';
    ctx.fillText(String(x0||0),pad.left,h-18);
    ctx.textAlign='right';
    ctx.fillText(String(x1||''),w-pad.right,h-18);

    const pts=series.map((d,i)=>({x:xAt(i),y:yAt(d.y)}));
    const segs=catmullRom2bezier(pts);
    canvas.__chartMeta={
      label:options.label||'指标',
      unit:options.unit||'',
      decimals:dec,
      plot:{left:pad.left,right:w-pad.right,top:pad.top,bottom:h-pad.bottom},
      points:pts.map((p,i)=>({
        x:p.x,
        y:p.y,
        step:series[i].x,
        value:series[i].y,
        time:series[i].time,
        lr:Number(series[i].lr),
        mem_gb:Number(series[i].mem_gb),
      })),
    };
    bindChartTooltip(canvas);

    function trace(pathPoints,closedToBottom){
      ctx.beginPath();
      ctx.moveTo(pathPoints[0].x,pathPoints[0].y);
      for(let i=0;i<segs.length;i++){
        const s=segs[i];
        ctx.bezierCurveTo(s[1].x,s[1].y,s[2].x,s[2].y,s[3].x,s[3].y);
      }
      if(closedToBottom){
        ctx.lineTo(pathPoints[pathPoints.length-1].x,h-pad.bottom);
        ctx.lineTo(pathPoints[0].x,h-pad.bottom);
        ctx.closePath();
      }
    }

    // watercolor wash under the curve
    trace(pts,true);
    const grad=ctx.createLinearGradient(0,pad.top,0,h-pad.bottom);
    grad.addColorStop(0,options.fill||'rgba(166,93,46,.22)');
    grad.addColorStop(1,'rgba(166,93,46,0)');
    ctx.fillStyle=grad;ctx.fill();

    // rolling average line
    if(options.rolling&&options.rolling.length>=2){
      const rpts=options.rolling.map((d,i)=>({x:xAt(i),y:yAt(d.y)}));
      const rsegs=catmullRom2bezier(rpts);
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(rpts[0].x,rpts[0].y);
      for(let i=0;i<rsegs.length;i++){
        const s=rsegs[i];
        ctx.bezierCurveTo(s[1].x,s[1].y,s[2].x,s[2].y,s[3].x,s[3].y);
      }
      ctx.strokeStyle=options.rollingColor||'rgba(46,41,30,.55)';
      ctx.lineWidth=1.6;
      ctx.setLineDash([4,3]);
      ctx.stroke();
      ctx.restore();
    }

    // main inked curve with soft shadow / registration offset
    ctx.save();
    ctx.translate(1.2,1.4);
    ctx.globalAlpha=.18;
    ctx.beginPath();
    ctx.moveTo(pts[0].x,pts[0].y);
    for(let i=0;i<segs.length;i++){const s=segs[i];ctx.bezierCurveTo(s[1].x,s[1].y,s[2].x,s[2].y,s[3].x,s[3].y);}
    ctx.strokeStyle='#2e291e';ctx.lineWidth=2;ctx.stroke();
    ctx.restore();

    ctx.beginPath();
    ctx.moveTo(pts[0].x,pts[0].y);
    for(let i=0;i<segs.length;i++){const s=segs[i];ctx.bezierCurveTo(s[1].x,s[1].y,s[2].x,s[2].y,s[3].x,s[3].y);}
    ctx.strokeStyle=options.color||'var(--copper)';ctx.lineWidth=2.4;ctx.lineCap='round';ctx.lineJoin='round';ctx.stroke();

    // specimen dots
    const skip=series.length>90?Math.ceil(series.length/50):1;
    pts.forEach((p,i)=>{
      if(i!==series.length-1 && i%skip!==0)return;
      ctx.beginPath();ctx.arc(p.x,p.y,2.4,0,Math.PI*2);ctx.fillStyle=options.color||'var(--copper)';ctx.fill();
      ctx.beginPath();ctx.arc(p.x,p.y,5.2,0,Math.PI*2);ctx.strokeStyle='rgba(46,41,30,.14)';ctx.lineWidth=1;ctx.stroke();
    });

    // callout: latest value, kept off the curve with a paper label and leader line.
    if(options.showAnnotations!==false){
      const last=series[series.length-1];
      const lastPt=pts[pts.length-1];
      ctx.save();
      const text=`最新 ${last.y.toFixed(dec)}`;
      ctx.font='11px Georgia,serif';
      const tw=ctx.measureText(text).width;
      const bw=tw+14,bh=20;
      const lx=Math.max(pad.left+6,Math.min(w-pad.right-bw-6,lastPt.x-bw-12));
      const preferAbove=lastPt.y>(pad.top+gh*.35);
      const rawY=preferAbove?lastPt.y-30:lastPt.y+12;
      const ly=Math.max(pad.top+6,Math.min(h-pad.bottom-bh-8,rawY));
      ctx.strokeStyle='rgba(92,83,64,.35)';
      ctx.lineWidth=1;
      ctx.beginPath();
      ctx.moveTo(lastPt.x,lastPt.y);
      ctx.lineTo(lx+bw,ly+bh/2);
      ctx.stroke();
      ctx.fillStyle='rgba(255,249,237,.92)';
      ctx.strokeStyle='rgba(166,93,46,.28)';
      ctx.lineWidth=1;
      ctx.beginPath();
      ctx.roundRect(lx,ly,bw,bh,6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle='#5c5340';
      ctx.textAlign='left';
      ctx.textBaseline='middle';
      ctx.fillText(text,lx+7,ly+bh/2+0.5);
      ctx.restore();
    }
  }catch(e){
    // do not let a single canvas error break the whole page
    try{const {ctx,w,h}=resizeCanvas(canvas);ctx.clearRect(0,0,w,h);ctx.fillStyle='#b53c2f';ctx.textAlign='center';ctx.fillText('图表渲染异常',w/2,h/2);}catch(_){}
  }
}

function setArcGauge(idPrefix,pct,color,innerText){
  const arc=$(idPrefix+'Arc');
  const text=$(idPrefix+'Text');
  if(!arc||!text)return;
  const r=44,circ=2*Math.PI*r;
  const v=Math.max(0,Math.min(100,pct||0));
  arc.style.strokeDasharray=`${circ}`;
  arc.style.strokeDashoffset=`${circ*(1-v/100)}`;
  if(color) arc.setAttribute('stroke',color);
  text.textContent=innerText!=null?String(innerText):'-';
}

let lastHistory=[];
let gradScaleMode='body';
function renderCharts(d){
  const hist=(d.history||[]).filter(r=>!r.type||r.type==='train');
  lastHistory=hist;
  const stats=d.history_stats||{};

  const lossSeries=hist.map(r=>({x:r.step,y:Number(r.loss),time:r.time,lr:r.lr,mem_gb:r.mem_gb})).filter(p=>Number.isFinite(p.y));
  const lossRolling=movingAvg(lossSeries,8);
  const maxStep=(d.progress&&d.progress.max_steps)||((hist[hist.length-1]||{}).max_steps)||0;
  const xEnd=((hist[hist.length-1]||{}).step)||maxStep||0;
  drawSmoothChart($('lossChart'),lossSeries,{label:'损失',color:'#a65d2e',fill:'rgba(166,93,46,.22)',decimals:4,rolling:lossRolling,rollingColor:'rgba(46,41,30,.5)',zeroBaseline:true,xStart:0,xEnd});
  const lossStats=stats.loss||{};
  $('lossChartNote').textContent=lossSeries.length>=2?`滚动均值 ${lossStats.rolling?lossStats.rolling.toFixed(4):'-'}`:'暂无足够历史数据';

  const toksSeries=hist.map(r=>({x:r.step,y:Number(r.tok_s),time:r.time,lr:r.lr,mem_gb:r.mem_gb})).filter(p=>Number.isFinite(p.y)&&p.y>0);
  const toksRolling=movingAvg(toksSeries,8);
  drawSmoothChart($('toksChart'),toksSeries,{label:'吞吐',unit:'Token/秒',color:'#4f7a5a',fill:'rgba(79,122,90,.22)',decimals:1,rolling:toksRolling,rollingColor:'rgba(46,41,30,.5)',zeroBaseline:true,xStart:0,xEnd});
  const toksStats=stats.tok_s||{};
  $('toksChartNote').textContent=toksSeries.length>=2?`平均 ${toksStats.avg?nf.format(toksStats.avg):'-'} Token/秒`:'暂无足够历史数据';

  const gradSeries=hist.map(r=>({x:r.step,y:Number(r.grad_norm),time:r.time,lr:r.lr,mem_gb:r.mem_gb})).filter(p=>Number.isFinite(p.y)&&p.y>0);
  const gradBodyMode=gradScaleMode!=='full';
  const gradDisplaySeries=gradBodyMode?filterRobustOutliers(gradSeries,.98):gradSeries;
  const gradRolling=movingAvg(gradDisplaySeries,8);
  drawSmoothChart($('gradChart'),gradDisplaySeries,{label:'裁剪前梯度范数',color:'#3b4f68',fill:'rgba(59,79,104,.18)',decimals:3,rolling:gradRolling,rollingColor:'rgba(46,41,30,.5)',zeroBaseline:true,xStart:0,xEnd});
  const gradStats=stats.grad_norm||{};
  $('gradChartNote').textContent=gradSeries.length>=2?`滚动均值 ${gradStats.rolling?gradStats.rolling.toFixed(3):'-'}`:'暂无足够历史数据';

  // GPU observatory gauges
  const g=d.gpu||{};
  const hasGpu=!!g.name;
  const util=hasGpu?(g.util_pct!=null?g.util_pct:0):0;
  const vramPct=hasGpu&&g.mem_total_mb?g.mem_used_mb/g.mem_total_mb*100:0;
  const temp=hasGpu?(g.temp_c!=null?g.temp_c:0):0;
  const tempPct=Math.min(100,temp/95*100);
  setArcGauge('gUtil',util,util>85?'var(--vermilion)':util>60?'var(--amber)':'var(--sage)',hasGpu?nfi.format(util)+'%':'未检测');
  setArcGauge('gVram',vramPct,vramPct>85?'var(--vermilion)':vramPct>60?'var(--amber)':'var(--copper)',hasGpu?nfi.format(vramPct)+'%':'未检测');
  setArcGauge('gTemp',tempPct,temp>85?'var(--vermilion)':temp>70?'var(--amber)':'var(--sage)',hasGpu?nfi.format(temp):'未检测');
  $('gPower').textContent=g.power_w!=null?nf.format(g.power_w)+' W':'未检测到';
  $('gName').textContent=g.name||'未检测到';
  $('gVramDetail').textContent=g.mem_total_mb?`${nf.format(g.mem_used_mb/1024)} / ${nf.format(g.mem_total_mb/1024)} GB`:'未检测到';
  $('gTempDetail').textContent=hasGpu?`${g.temp_c!=null?nf.format(g.temp_c)+' °C':'-'} · ${g.power_w!=null?nf.format(g.power_w)+' W':'-'}`:'未检测到';
  $('gTokDetail').textContent=(d.train&&d.train.tok_s)?`${nf.format(d.train.tok_s)} Token/秒`:'暂无吞吐';
  $('gObsNote').textContent=hasGpu?`${util>=95?'高负载稳定采样':util>=70?'负载良好':'负载偏低'} · 显存 ${nf.format(vramPct)}%`:'等待 GPU 采样';
  $('gObsTime').textContent=dt(d.time);
}

async function refresh(){
  let d={};
  try{
    d=await (await fetch('/api/status',{cache:'no-store'})).json();
  }catch(e){
    $('updateTime').textContent='拉取失败 '+new Date().toLocaleTimeString('zh-CN');
    return;
  }
  const tr=d.train||{}, p=d.progress||{}, g=d.gpu||{}, st=d.start||{};
  const sm=stateMeta(d.state);
  const hm=healthMeta(d.health);

  $('run').textContent=d.run_dir||'-';
  const badge=$('stateBadge');
  badge.className='badge '+sm.dot+'-bg';
  badge.innerHTML=`<span class="dot ${sm.dot} ${d.state==='running'?'pulse':''}"></span><span>${sm.text}</span>`;
  const hb=$('healthBadge');
  hb.className='badge '+hm.dot+'-bg';
  const reasons=(d.health&&d.health.reasons)||[];
  hb.innerHTML=`<span class="dot ${hm.dot}"></span><span title="${reasons.join(' | ')}">${hm.text}</span>`;

  const pids=(d.pids||[]);
  $('pidLine').innerHTML='进程：'+(pids.length?`<b>${pids.join(', ')}</b>`:'<b class="missing">未检测到</b>');
  $('updateTime').textContent=dt(d.time);

  const step=p.step||0, max=p.max_steps||0, pct=max?step/max*100:0;
  $('pct').textContent=Number.isFinite(pct)?nf.format(pct)+'%':'-';
  $('fill').style.width=Math.max(0,Math.min(100,pct))+'%';
  $('progressSub').textContent=max?`步数 ${nf.format(step)} / ${nf.format(max)} · 每次更新 ${fmtTokens(p.tokens_per_update)} Token`:'暂无训练进度数据';
  $('step').textContent=(step===0&&max===0)?'暂无':nf.format(step);
  $('remainSteps').textContent=max?nf.format(p.remain_steps):'暂无';
  $('trainedTokens').textContent=fmtTokens(p.trained_tokens);
  $('remainTokens').textContent=max?fmtTokens(p.remain_tokens):'暂无';
  $('eta').textContent=dur(p.eta_s);
  $('etaDetail').textContent='';

  const vramPct=g.mem_total_mb?(g.mem_used_mb/g.mem_total_mb*100):0;

  $('events').innerHTML=(d.events||[]).map(x=>`<li>${x}</li>`).join('');

  const ckpts=d.checkpoints||[];
  $('ckpts').innerHTML=ckpts.length?ckpts.map(x=>`<tr><td>${x.name}</td><td>${nf.format(x.gb)} GB</td><td>${dt(x.mtime)}</td></tr>`).join(''):'<tr><td colspan="3" class="empty">暂无检查点</td></tr>';

  window.__lastData=d;
  renderCharts(d);
}

window.addEventListener('resize',()=>{renderCharts(window.__lastData||{});});
document.addEventListener('click',event=>{
  const btn=event.target&&event.target.closest?event.target.closest('[data-grad-scale]'):null;
  if(!btn)return;
  gradScaleMode=btn.dataset.gradScale==='full'?'full':'body';
  document.querySelectorAll('[data-grad-scale]').forEach(x=>x.classList.toggle('active',x===btn));
  renderCharts(window.__lastData||{});
});
refresh();
setInterval(refresh,5000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    run_dir = DEFAULT_RUN_DIR
    launch_log = Path(f"{DEFAULT_RUN_DIR}.launch.log")

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def send_body(self, body: bytes, content_type: str, code: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(int(code))
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_body(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            payload = status(self.run_dir, self.launch_log)
            self.send_body(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        elif self.path == "/healthz":
            self.send_body(b"ok\n", "text/plain; charset=utf-8")
        else:
            self.send_body(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--launch-log", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=16006)
    args = parser.parse_args()

    Handler.run_dir = args.run_dir
    Handler.launch_log = args.launch_log or Path(f"{args.run_dir}.launch.log")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "dashboard_started", "url": f"http://{args.host}:{args.port}/"}, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
