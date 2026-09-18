#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SOPHIA_PYTHON:-python3}"
DATA_PATH="${SOPHIA_DATA_PATH:-}"
TOKENIZER_PATH="${SOPHIA_TOKENIZER_PATH:-ml/modeling/text}"
OUTPUT_ROOT="${SOPHIA_TARGET_PREP_ROOT:-$ROOT/ops/pretrain_target_prep}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

fail() { echo "FATAL: $*" >&2; exit 1; }

[ -n "$DATA_PATH" ] || fail "SOPHIA_DATA_PATH must point to fresh production train shards"
[ -f "$DATA_PATH/dataset_manifest.json" ] && DATA_PATH="${DATA_PATH%/}/train"
[ -f "$DATA_PATH/manifest.json" ] || fail "missing train manifest: $DATA_PATH/manifest.json"
[ -f "$TOKENIZER_PATH/tokenizer.json" ] || fail "missing tokenizer: $TOKENIZER_PATH/tokenizer.json"
[ ! -e "$OUTPUT_ROOT" ] || fail "target prep output already exists; choose a fresh SOPHIA_TARGET_PREP_ROOT"
mkdir -p "$OUTPUT_ROOT"

run_selected_muon() {
    local recipe_root="$OUTPUT_ROOT/recipe_measurement_muon"
    local recipe="$recipe_root/release_pretrain_machine_recipe.json"
    local probe_dir="$OUTPUT_ROOT/stability_probe_muon"
    local resume_probe_dir="$OUTPUT_ROOT/resume_probe_muon"
    local recapture_probe_dir="$OUTPUT_ROOT/graph_recapture_probe_muon"
    local warmup_checkpoint="$probe_dir/checkpoints/ckpt_step153.pt"

    "$PYTHON_BIN" -m ml.tooling.scripts.machine_recipes.check_pretrain_machine_recipe \
        --data_path "$DATA_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --output_dir "$recipe_root"

    [ -f "$recipe" ] || fail "signed recipe was not produced: $recipe"

    "$PYTHON_BIN" -m ml.tooling.scripts.pretrain_stability_probe \
        --data_path "$DATA_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --machine_recipe_json "$recipe" \
        --output_dir "$recapture_probe_dir" \
        --output_json "$OUTPUT_ROOT/graph_recapture_probe.json" \
        --graph_recapture_probe 1 \
        --max_steps 12 \
        --eval_interval 10 \
        --eval_steps 1 \
        --checkpoint_interval 0 \
        --checkpoint_keep 1

    "$PYTHON_BIN" -m ml.tooling.scripts.pretrain_stability_probe \
        --data_path "$DATA_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --machine_recipe_json "$recipe" \
        --output_dir "$probe_dir" \
        --output_json "$OUTPUT_ROOT/muon_stability.json" \
        --max_steps 154 \
        --eval_interval 154 \
        --eval_steps 1 \
        --checkpoint_interval 153 \
        --checkpoint_keep 2

    [ -f "$warmup_checkpoint" ] || fail "missing warmup-boundary checkpoint: $warmup_checkpoint"
    [ -f "$warmup_checkpoint.sha256" ] || fail "missing warmup-boundary checkpoint digest: $warmup_checkpoint.sha256"

    "$PYTHON_BIN" -m ml.tooling.scripts.pretrain_stability_probe \
        --data_path "$DATA_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --machine_recipe_json "$recipe" \
        --output_dir "$resume_probe_dir" \
        --output_json "$OUTPUT_ROOT/muon_stability_resume.json" \
        --resume_from_checkpoint "$warmup_checkpoint" \
        --max_steps 154 \
        --eval_interval 154 \
        --eval_steps 1 \
        --checkpoint_interval 153 \
        --checkpoint_keep 2
}

run_selected_muon

"$PYTHON_BIN" -m ml.tooling.scripts.audit_selected_muon_probe \
    --recipe "$OUTPUT_ROOT/recipe_measurement_muon/release_pretrain_machine_recipe.json" \
    --stability "$OUTPUT_ROOT/muon_stability.json" \
    --resume "$OUTPUT_ROOT/muon_stability_resume.json" \
    --recapture "$OUTPUT_ROOT/graph_recapture_probe.json" \
    --output_json "$OUTPUT_ROOT/muon_probe_audit.json"

echo "selected Muon recipe and recapture/stability/resume validation complete: $OUTPUT_ROOT"
echo "formal recipe: $OUTPUT_ROOT/recipe_measurement_muon/release_pretrain_machine_recipe.json"
echo "formal audit:  $OUTPUT_ROOT/muon_probe_audit.json"
