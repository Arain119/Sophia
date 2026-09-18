#!/usr/bin/env bash
# SFT launcher: validate admitted dataset + parent checkpoint, dry-run the
# trainer, then launch it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SOPHIA_PYTHON:-python3}"
DATASET_DIR="${SOPHIA_SFT_DATASET_DIR:-release/corpus}"
PARENT_CHECKPOINT="${SOPHIA_SFT_PARENT_CHECKPOINT:?SOPHIA_SFT_PARENT_CHECKPOINT is required}"
OUTPUT_DIR="${SOPHIA_SFT_OUTPUT_DIR:-runs/sft}"
TOKENIZER_PATH="${SOPHIA_SFT_TOKENIZER_PATH:-ml/modeling/text}"
MODEL_SPEC="${SOPHIA_SFT_MODEL_SPEC:-configs/model/sophia.json}"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

SOPHIA_CHECK_DATASET_DIR="$DATASET_DIR" \
SOPHIA_CHECK_PARENT_CHECKPOINT="$PARENT_CHECKPOINT" \
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from ml.training.sft.base_admission import load_and_validate_parent_checkpoint

dataset = Path(os.environ["SOPHIA_CHECK_DATASET_DIR"])
for name in ("train.jsonl", "val.jsonl"):
    if not (dataset / name).is_file():
        raise SystemExit(f"missing {dataset / name}")
load_and_validate_parent_checkpoint(
    Path(os.environ["SOPHIA_CHECK_PARENT_CHECKPOINT"])
)
PY

CMD=("$PYTHON_BIN" -m ml.cli.sft_train
    --dataset "$DATASET_DIR"
    --output "$OUTPUT_DIR"
    --parent "$PARENT_CHECKPOINT"
    --tokenizer-path "$TOKENIZER_PATH"
    --model-spec "$MODEL_SPEC")

"${CMD[@]}" --dry-run

if [ "$DRY_RUN" = "1" ]; then
    printf '  %q' "${CMD[@]}"; echo
    exit 0
fi

[ ! -e "$OUTPUT_DIR" ] || [ -z "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] || {
    echo "FATAL: output directory is not empty: $OUTPUT_DIR" >&2; exit 1; }
exec "${CMD[@]}"
