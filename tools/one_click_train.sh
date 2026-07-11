#!/usr/bin/env bash
# One-click training launcher (pretrain + SFT).
#
# Runs the full gate chain (environment -> data assets -> regression subset ->
# GPU probe) and then launches the requested training stage as a detached
# background process with its launch log outside the run output directory.
#
# Usage:
#   bash tools/one_click_train.sh [pretrain|sft] [--dry-run]
#     stage      defaults to pretrain
#     --dry-run  run every gate, print the launch command, skip the launch
#
# Environment overrides (all optional):
#   SOPHIA_PYTHON           python interpreter          (default: python3)
#   SOPHIA_OUTPUT_DIR       run output directory        (default: runs/pretrain_rtx5090_bf16
#                                                        or runs/sft_rtx5090_bf16)
#   SOPHIA_RESUME           auto|latest|<ckpt path>     (default: empty = fresh run)
#   SOPHIA_SKIP_TESTS=1     skip the regression subset
#   SOPHIA_SKIP_PROBE=1     skip the GPU readiness probe
#
# pretrain stage:
#   SOPHIA_DATA_PATH        train split or dataset root (default: dataset/pretrain_tokens/train)
#   SOPHIA_DECAY_DATA_PATH  WSD decay-phase dataset     (default: dataset/pretrain_decay/train;
#                                                        auto-dropped when missing)
#   SOPHIA_TOKENIZER_PATH   tokenizer bundle            (default: ml/modeling/text)
#   SOPHIA_MACHINE_RECIPE   signed machine recipe JSON  (default: canonical recipe)
#
# sft stage:
#   SOPHIA_EXPORT_DIR       pretrain export directory   (default: runs/pretrain_rtx5090_bf16)
#   SOPHIA_SFT_TRAIN_DATA   chat jsonl train split      (default: dataset/sft/train.jsonl)
#   SOPHIA_SFT_EVAL_DATA    chat jsonl eval split       (default: dataset/sft/val.jsonl)
#   SOPHIA_SFT_TEST_DATA    chat jsonl test split       (default: dataset/sft/test.jsonl)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SOPHIA_PYTHON:-python3}"
RESUME="${SOPHIA_RESUME:-}"

STAGE="pretrain"
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        pretrain|sft) STAGE="$arg" ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

fail() { echo "FATAL: $*" >&2; exit 1; }

echo "[1/5] environment (stage=$STAGE)"
"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit(
        f"FATAL: python {sys.version.split()[0]} < 3.12; "
        "point SOPHIA_PYTHON at the project venv interpreter."
    )

import torch

print(f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is not available; release training requires a CUDA GPU.")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "[2/5] data assets"
resolve_root() {
    # Accept either a dataset root or a train/ split directory.
    case "$1" in *"/train"|*"/train/") dirname "$1" ;; *) echo "$1" ;; esac
}
if [ "$STAGE" = "pretrain" ]; then
    DATA_PATH="${SOPHIA_DATA_PATH:-dataset/pretrain_tokens/train}"
    DECAY_DATA_PATH="${SOPHIA_DECAY_DATA_PATH:-dataset/pretrain_decay/train}"
    TOKENIZER_PATH="${SOPHIA_TOKENIZER_PATH:-ml/modeling/text}"
    OUTPUT_DIR="${SOPHIA_OUTPUT_DIR:-runs/pretrain_rtx5090_bf16}"
    MACHINE_RECIPE="${SOPHIA_MACHINE_RECIPE:-}"

    DATA_ROOT="$(resolve_root "$DATA_PATH")"
    for split in train val test; do
        [ -f "$DATA_ROOT/$split/manifest.json" ] \
            || fail "missing $DATA_ROOT/$split/manifest.json (build token shards first: ml-shard)"
    done
    [ -f "$TOKENIZER_PATH/tokenizer.json" ] || fail "missing tokenizer bundle at $TOKENIZER_PATH"
    if [ -n "$DECAY_DATA_PATH" ]; then
        DECAY_ROOT="$(resolve_root "$DECAY_DATA_PATH")"
        if [ ! -f "$DECAY_ROOT/train/manifest.json" ]; then
            echo "note: decay dataset not found at $DECAY_DATA_PATH; training on --data_path for the whole run"
            DECAY_DATA_PATH=""
        fi
    fi
    echo "data=$DATA_PATH decay=${DECAY_DATA_PATH:-<none>} tokenizer=$TOKENIZER_PATH"
else
    EXPORT_DIR="${SOPHIA_EXPORT_DIR:-runs/pretrain_rtx5090_bf16}"
    SFT_TRAIN_DATA="${SOPHIA_SFT_TRAIN_DATA:-dataset/sft/train.jsonl}"
    SFT_EVAL_DATA="${SOPHIA_SFT_EVAL_DATA:-dataset/sft/val.jsonl}"
    SFT_TEST_DATA="${SOPHIA_SFT_TEST_DATA:-dataset/sft/test.jsonl}"
    OUTPUT_DIR="${SOPHIA_OUTPUT_DIR:-runs/sft_rtx5090_bf16}"

    for name in config.json tokenizer.json; do
        [ -f "$EXPORT_DIR/$name" ] \
            || fail "missing $EXPORT_DIR/$name (run the pretrain stage first; its export lands in the pretrain output_dir)"
    done
    ls "$EXPORT_DIR"/*.safetensors >/dev/null 2>&1 || ls "$EXPORT_DIR"/pytorch_model*.bin >/dev/null 2>&1 \
        || fail "no model weights found under $EXPORT_DIR"
    [ -f "$SFT_TRAIN_DATA" ] || fail "missing SFT train data at $SFT_TRAIN_DATA"
    [ -f "$SFT_EVAL_DATA" ] || { echo "note: eval split not found at $SFT_EVAL_DATA"; SFT_EVAL_DATA=""; }
    [ -f "$SFT_TEST_DATA" ] || { echo "note: test split not found at $SFT_TEST_DATA"; SFT_TEST_DATA=""; }
    [ -f "dataset/posttrain_length_curriculum.json" ] \
        || fail "missing dataset/posttrain_length_curriculum.json (regenerate: python3 -m ml.tooling.scripts.data.production.write_posttrain_length_curriculum --sft_dir dataset/sft --output dataset/posttrain_length_curriculum.json)"
    echo "export=$EXPORT_DIR train=$SFT_TRAIN_DATA eval=${SFT_EVAL_DATA:-<none>} test=${SFT_TEST_DATA:-<none>}"
fi

echo "[3/5] regression subset"
if [ "${SOPHIA_SKIP_TESTS:-0}" = "1" ]; then
    echo "skipped (SOPHIA_SKIP_TESTS=1)"
else
    "$PYTHON_BIN" -m pytest -q \
        tests/tests/test_config_canonical.py \
        tests/tests/test_pretrain_smoke.py \
        tests/tests/test_cli_entrypoints.py \
        tests/tests/test_runtime_import_boundary.py
fi

echo "[4/5] GPU readiness probe"
if [ "${SOPHIA_SKIP_PROBE:-0}" = "1" ]; then
    echo "skipped (SOPHIA_SKIP_PROBE=1)"
else
    SOPHIA_REMOTE_ROOT="${SOPHIA_REMOTE_ROOT:-$ROOT/ops}" \
    SOPHIA_PYTHON="$PYTHON_BIN" \
        bash tools/pretrain_gpu_probe.sh
fi

echo "[5/5] launch"
if [ -e "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] && [ -z "$RESUME" ]; then
    fail "output_dir $OUTPUT_DIR is not empty; set SOPHIA_RESUME=auto to resume or pick a fresh SOPHIA_OUTPUT_DIR"
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
LAUNCH_LOG="${OUTPUT_DIR%/}.launch.log"
PID_FILE="${OUTPUT_DIR%/}.pid"

if [ "$STAGE" = "pretrain" ]; then
    CMD=("$PYTHON_BIN" -m ml.cli.train
        --data_path "$DATA_PATH"
        --tokenizer_path "$TOKENIZER_PATH"
        --output_dir "$OUTPUT_DIR")
    [ -n "$DECAY_DATA_PATH" ] && CMD+=(--decay_data_path "$DECAY_DATA_PATH")
    [ -n "$MACHINE_RECIPE" ] && CMD+=(--machine_recipe_json "$MACHINE_RECIPE")
else
    CMD=("$PYTHON_BIN" -m ml.cli.sft
        --export_dir "$EXPORT_DIR"
        --train_data "$SFT_TRAIN_DATA"
        --output_dir "$OUTPUT_DIR")
    [ -n "$SFT_EVAL_DATA" ] && CMD+=(--eval_data "$SFT_EVAL_DATA")
    [ -n "$SFT_TEST_DATA" ] && CMD+=(--test_data "$SFT_TEST_DATA")
fi
[ -n "$RESUME" ] && CMD+=(--resume_from_checkpoint "$RESUME")

if [ "$DRY_RUN" = "1" ]; then
    echo "dry run; would launch:"
    printf '  %q' "${CMD[@]}"; echo
    exit 0
fi

nohup "${CMD[@]}" > "$LAUNCH_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

sleep 30
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "---- last launch log lines ----"
    tail -n 40 "$LAUNCH_LOG"
    fail "training process exited within 30s; see $LAUNCH_LOG"
fi

echo "launched: stage=$STAGE pid=$TRAIN_PID"
echo "launch log:  $LAUNCH_LOG"
echo "monitor:     tail -f $LAUNCH_LOG"
echo "dashboard:   $PYTHON_BIN tools/live_train_dashboard.py --run-dir $OUTPUT_DIR"
echo "loop check:  $PYTHON_BIN tools/train_monitor_loop.py --run-dir $OUTPUT_DIR --launch-log $LAUNCH_LOG"
