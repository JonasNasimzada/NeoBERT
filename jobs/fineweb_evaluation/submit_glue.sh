#!/usr/bin/env bash
# Submit the pinned-data, A100-preflight, full, and aggregate GLUE chain.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEOBERT_ROOT="${NEOBERT_ROOT:-$(cd "$script_dir/../.." && pwd)}"
COMPLEX_ATTENTION_ROOT="${COMPLEX_ATTENTION_ROOT:-$(cd "$NEOBERT_ROOT/.." && pwd)}"
GLUE_OUTPUT_ROOT="${GLUE_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/glue-validation-v1}"
GLUE_DATA_ROOT="${GLUE_DATA_ROOT:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/glue/nyu-mll-glue-bcdcba79}"
export NEOBERT_ROOT COMPLEX_ATTENTION_ROOT GLUE_OUTPUT_ROOT GLUE_DATA_ROOT

mkdir -p "$GLUE_OUTPUT_ROOT/slurm"
python3 "$NEOBERT_ROOT/scripts/fineweb_evaluation/glue_prepare_data.py" \
    --output "$GLUE_DATA_ROOT"

preflight_id="$(
    sbatch --parsable \
        --output="$GLUE_OUTPUT_ROOT/slurm/preflight-%j.out" \
        "$script_dir/glue_preflight.sbatch"
)"
full_id="$(
    sbatch --parsable \
        --dependency="afterok:$preflight_id" \
        --output="$GLUE_OUTPUT_ROOT/slurm/full-%A_%a.out" \
        "$script_dir/glue.sbatch"
)"
aggregate_id="$(
    sbatch --parsable \
        --dependency="afterok:$full_id" \
        --output="$GLUE_OUTPUT_ROOT/slurm/aggregate-%j.out" \
        "$script_dir/glue_aggregate.sbatch"
)"

printf 'GLUE data: %s\n' "$GLUE_DATA_ROOT"
printf 'GLUE preflight job: %s\n' "$preflight_id"
printf 'GLUE full array job: %s (afterok:%s)\n' "$full_id" "$preflight_id"
printf 'GLUE aggregate job: %s (afterok:%s)\n' "$aggregate_id" "$full_id"
