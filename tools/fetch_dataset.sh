#!/usr/bin/env bash
# Fetch the public Sophia training dataset from ModelScope.
#
# Source: https://www.modelscope.cn/datasets/Arain119/Sophia-dataset
#   pretrain_tokens/  tokenized pretrain shards (train/val/test)   ~62 GB
#   pretrain_decay/   WSD decay-phase shards (train)               ~9 GB
#   sft/              chat JSONL splits (train/val/test)           ~22 MB
#
# The download is staged deliberately: split manifests come down first and the
# tokenizer fingerprint is verified against the local bundle BEFORE any
# multi-gigabyte shard transfer starts. A fingerprint mismatch would be
# rejected by the training preflight anyway; failing here saves the bandwidth.
#
# Usage:
#   bash tools/fetch_dataset.sh [pretrain|sft|all] [--manifests-only]
#     stage             defaults to all (pretrain + sft)
#     --manifests-only  download split manifests and run the fingerprint /
#                       layout checks only; skip the shard payload
#
# Environment overrides (all optional):
#   SOPHIA_PYTHON          python interpreter        (default: python3)
#   SOPHIA_DATASET_ID      ModelScope dataset id     (default: Arain119/Sophia-dataset)
#   SOPHIA_DATASET_DIR     local destination root    (default: dataset)
#   SOPHIA_TOKENIZER_PATH  tokenizer bundle to check (default: ml/modeling/text)
#
# Downloads are resumable: rerun the same command after an interruption.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SOPHIA_PYTHON:-python3}"
DATASET_ID="${SOPHIA_DATASET_ID:-Arain119/Sophia-dataset}"
DEST="${SOPHIA_DATASET_DIR:-dataset}"
TOKENIZER_PATH="${SOPHIA_TOKENIZER_PATH:-ml/modeling/text}"

STAGE="all"
MANIFESTS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        pretrain|sft|all) STAGE="$arg" ;;
        --manifests-only) MANIFESTS_ONLY=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$ROOT"
fail() { echo "FATAL: $*" >&2; exit 1; }

echo "[1/4] modelscope CLI"
find_modelscope() {
    local candidate
    candidate="$(dirname "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')")/modelscope"
    if [ -x "$candidate" ]; then echo "$candidate"; return 0; fi
    command -v modelscope 2>/dev/null || true
}
MS_BIN="$(find_modelscope)"
if [ -z "$MS_BIN" ]; then
    echo "modelscope not installed; installing into the current interpreter"
    "$PYTHON_BIN" -m pip install -q -U modelscope
    MS_BIN="$(find_modelscope)"
fi
[ -n "$MS_BIN" ] || fail "modelscope CLI not found after install; run: $PYTHON_BIN -m pip install modelscope"
echo "modelscope=$MS_BIN dataset=$DATASET_ID dest=$DEST stage=$STAGE"

ms_download() {
    "$MS_BIN" download --dataset "$DATASET_ID" --local_dir "$DEST" --include "$@"
}

echo "[2/4] split manifests + fingerprint check"
if [ "$STAGE" = "sft" ]; then
    echo "skipped (sft data carries no token-shard manifests)"
else
    ms_download 'pretrain_tokens/*/manifest.json' 'pretrain_decay/*/manifest.json'
    SOPHIA_FETCH_DEST="$DEST" SOPHIA_FETCH_TOKENIZER="$TOKENIZER_PATH" "$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import sys

root = os.getcwd()
dest = os.environ["SOPHIA_FETCH_DEST"]
tokenizer_dir = os.environ["SOPHIA_FETCH_TOKENIZER"]

spec = importlib.util.spec_from_file_location(
    "tokenizer_fingerprint",
    os.path.join(root, "ml", "data", "token_shards", "tokenizer_fingerprint.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

local_sha1 = mod.compute_tokenizer_bundle_sha1(tokenizer_dir).lower()
print(f"local tokenizer bundle: {tokenizer_dir} sha1={local_sha1}")

manifests = [
    os.path.join(dest, "pretrain_tokens", split, "manifest.json")
    for split in ("train", "val", "test")
]
manifests.append(os.path.join(dest, "pretrain_decay", "train", "manifest.json"))

mismatched = []
for path in manifests:
    if not os.path.isfile(path):
        print(f"note: {path} not present in the dataset download; skipping")
        continue
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    expected = str(manifest.get("tokenizer_sha1", "")).strip().lower()
    status = "OK" if expected == local_sha1 else "MISMATCH"
    print(f"{status}: {path} tokenizer_sha1={expected}")
    if expected != local_sha1:
        mismatched.append((path, expected))

if mismatched:
    sys.exit(
        "FATAL: dataset manifests were built with a different tokenizer bundle.\n"
        + "\n".join(f"  {p}: manifest={e}" for p, e in mismatched)
        + f"\n  local bundle ({tokenizer_dir}): {local_sha1}\n"
        "The training preflight will reject this combination. Do NOT edit the\n"
        "manifests or bypass the check. Align the versions instead: use the\n"
        "tokenizer bundle the dataset was built with (ask the dataset publisher\n"
        "to ship it alongside the shards), or a dataset release built with the\n"
        "repository bundle."
    )
print("fingerprint check passed")
PY
fi

if [ "$MANIFESTS_ONLY" = "1" ]; then
    echo "[3/4] payload download skipped (--manifests-only)"
    echo "[4/4] done"
    exit 0
fi

echo "[3/4] payload download"
case "$STAGE" in
    pretrain) ms_download 'pretrain_tokens/*' 'pretrain_decay/*' ;;
    sft)      ms_download 'sft/*' ;;
    all)      ms_download 'pretrain_tokens/*' 'pretrain_decay/*' 'sft/*' ;;
esac

echo "[4/4] integrity check"
if [ "$STAGE" != "sft" ]; then
    SOPHIA_FETCH_DEST="$DEST" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

dest = os.environ["SOPHIA_FETCH_DEST"]
ITEM_BYTES = 4  # canonical int32 shard format

roots = [
    (os.path.join(dest, "pretrain_tokens"), ("train", "val", "test")),
    (os.path.join(dest, "pretrain_decay"), ("train",)),
]
problems = []
for root, splits in roots:
    for split in splits:
        manifest_path = os.path.join(root, split, "manifest.json")
        if not os.path.isfile(manifest_path):
            problems.append(f"missing manifest: {manifest_path}")
            continue
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        total = 0
        for shard in manifest.get("shards", []):
            path = os.path.join(root, split, shard["path"])
            expected = int(shard["tokens"]) * ITEM_BYTES
            if not os.path.isfile(path):
                problems.append(f"missing shard: {path}")
                continue
            actual = os.path.getsize(path)
            if actual != expected:
                problems.append(
                    f"size mismatch: {path} expected={expected} actual={actual}"
                    " (interrupted download? rerun this script to resume)"
                )
            total += int(shard["tokens"])
        print(
            f"{root}/{split}: shards={len(manifest.get('shards', []))} "
            f"tokens={total:,} manifest_total={int(manifest.get('total_tokens', 0)):,}"
        )
if problems:
    sys.exit("FATAL: dataset integrity check failed:\n" + "\n".join(f"  {p}" for p in problems))
print("shard integrity check passed")
PY
fi
if [ "$STAGE" != "pretrain" ]; then
    for split in train val test; do
        f="$DEST/sft/$split.jsonl"
        if [ -f "$f" ]; then
            "$PYTHON_BIN" -c "
import json, sys
path = '$f'
with open(path, encoding='utf-8') as fh:
    n = 0
    for line in fh:
        if line.strip():
            json.loads(line)
            n += 1
print(f'{path}: {n} samples')
"
        else
            echo "note: $f not present in the dataset"
        fi
    done
fi

echo "done. next: bash tools/one_click_train.sh pretrain --dry-run"
