#!/usr/bin/env bash
# Pretrain launcher: environment -> regression subset -> GPU probe -> detached launch.
#
#   bash tools/one_click_train.sh [--dry-run]
#
#   SOPHIA_DATA_PATH      fresh train split or dataset root (required)
#   SOPHIA_OUTPUT_DIR     run output directory (default: runs/pretrain_rtx5090_bf16)
#   SOPHIA_RESUME         auto|latest|<ckpt path> (default: fresh run)
#   SOPHIA_SKIP_TESTS=1   skip the regression subset
#   SOPHIA_SKIP_PROBE=1   skip the GPU readiness probe
#   SOPHIA_TOKENIZER_PATH tokenizer bundle (default: ml/modeling/text)
#   SOPHIA_MACHINE_RECIPE            signed machine recipe JSON (required)
#   SOPHIA_MUON_PROBE_AUDIT          probe audit JSON for that recipe (required)
#   SOPHIA_MACHINE_RECIPE_MIGRATION_FROM_SHA256   old recipe SHA for explicit
#                                               resume migration (optional)
#   SOPHIA_ALLOW_PROTOCOL_PATH_RELOCATION=1       allow canonical protocol path
#                                               relocation on resume (optional)
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SOPHIA_PYTHON:-python3}"
RESUME="${SOPHIA_RESUME:-}"
SOPHIA_DATA_PATH="${SOPHIA_DATA_PATH:?SOPHIA_DATA_PATH is required (fresh shards)}"
MACHINE_RECIPE="${SOPHIA_MACHINE_RECIPE:?SOPHIA_MACHINE_RECIPE is required (signed recipe JSON)}"
MUON_PROBE_AUDIT="${SOPHIA_MUON_PROBE_AUDIT:?SOPHIA_MUON_PROBE_AUDIT is required}"
MACHINE_RECIPE_MIGRATION_FROM_SHA256="${SOPHIA_MACHINE_RECIPE_MIGRATION_FROM_SHA256:-}"
ALLOW_PROTOCOL_PATH_RELOCATION="${SOPHIA_ALLOW_PROTOCOL_PATH_RELOCATION:-0}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
fail() { echo "FATAL: $*" >&2; exit 1; }

"$PYTHON_BIN" - <<'PY'
import sys
import torch
print(f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda}")
assert sys.version_info >= (3, 12), "python >= 3.12 required"
assert torch.cuda.is_available(), "CUDA GPU required"
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

TOKENIZER_PATH="${SOPHIA_TOKENIZER_PATH:-ml/modeling/text}"
OUTPUT_DIR="${SOPHIA_OUTPUT_DIR:-runs/pretrain_rtx5090_bf16}"
echo "data=$SOPHIA_DATA_PATH tokenizer=$TOKENIZER_PATH"

# Checkpoints must land on target-native storage.
probe_path="$OUTPUT_DIR"
while [ ! -e "$probe_path" ] && [ "$probe_path" != "/" ]; do
    probe_path="$(dirname "$probe_path")"
done
if command -v findmnt >/dev/null 2>&1; then
    case "$(findmnt -n -o FSTYPE --target "$probe_path" 2>/dev/null || true)" in
        9p|drvfs|fuseblk)
            fail "output must use target-native storage, not a host-mounted filesystem: $OUTPUT_DIR" ;;
    esac
fi

if [ "${SOPHIA_SKIP_TESTS:-0}" != "1" ]; then
    "$PYTHON_BIN" -m pytest -q \
        tests/tests/test_config_canonical.py \
        tests/tests/test_pretrain_smoke.py \
        tests/tests/test_cli_entrypoints.py \
        tests/tests/test_runtime_import_boundary.py
fi

if [ "${SOPHIA_SKIP_PROBE:-0}" != "1" ]; then
    SOPHIA_REMOTE_ROOT="${SOPHIA_REMOTE_ROOT:-$ROOT/ops}" \
    SOPHIA_PYTHON="$PYTHON_BIN" \
        bash tools/pretrain_gpu_probe.sh
fi

[ ! -e "$OUTPUT_DIR" ] || [ -z "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] || [ -n "$RESUME" ] || \
    fail "output_dir $OUTPUT_DIR is not empty; set SOPHIA_RESUME=auto or pick a fresh SOPHIA_OUTPUT_DIR"

OUTPUT_PARENT="$(dirname "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_PARENT"
LAUNCH_LOG="${OUTPUT_DIR%/}.launch.log"
PID_FILE="${OUTPUT_DIR%/}.pid"

CMD=("$PYTHON_BIN" -m ml.cli.train
    --data_path "$SOPHIA_DATA_PATH"
    --tokenizer_path "$TOKENIZER_PATH"
    --output_dir "$OUTPUT_DIR"
    --machine_recipe_json "$MACHINE_RECIPE"
    --muon_probe_audit_json "$MUON_PROBE_AUDIT")
[ -n "$MACHINE_RECIPE_MIGRATION_FROM_SHA256" ] \
    && CMD+=(--machine_recipe_migration_from_sha256 "$MACHINE_RECIPE_MIGRATION_FROM_SHA256")
[ "$ALLOW_PROTOCOL_PATH_RELOCATION" = "1" ] \
    && CMD+=(--allow_protocol_path_relocation 1)
[ -n "$RESUME" ] && CMD+=(--resume_from_checkpoint "$RESUME")

# Preflight in a temp dir on the same filesystem: the release gate checks free
# space through output_dir.
PREFLIGHT_DIR="$(mktemp -d "$OUTPUT_PARENT/.sophia-one-click-preflight.XXXXXX")"
trap 'rm -rf -- "$PREFLIGHT_DIR"' EXIT
PREFLIGHT_CMD=("${CMD[@]}")
for ((i = 0; i < ${#PREFLIGHT_CMD[@]}; i++)); do
    if [ "${PREFLIGHT_CMD[$i]}" = "--output_dir" ]; then
        PREFLIGHT_CMD[$((i + 1))]="$PREFLIGHT_DIR"
        break
    fi
done
"${PREFLIGHT_CMD[@]}" --preflight 1
rm -rf -- "$PREFLIGHT_DIR"
trap - EXIT

if [ "$DRY_RUN" = "1" ]; then
    printf '  %q' "${CMD[@]}"; echo
    exit 0
fi

nohup "${CMD[@]}" > "$LAUNCH_LOG" 2>&1 &
echo "$!" > "$PID_FILE"
sleep 30
kill -0 "$!" 2>/dev/null || { tail -n 40 "$LAUNCH_LOG"; fail "training exited within 30s; see $LAUNCH_LOG"; }
echo "launched: pid=$(cat "$PID_FILE") log=$LAUNCH_LOG"
