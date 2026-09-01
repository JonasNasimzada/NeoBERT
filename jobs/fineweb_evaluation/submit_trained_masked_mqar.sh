#!/usr/bin/env bash
# Submit A100 preflight, pilot, two replication seeds, and final comparison.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEOBERT_ROOT="$(cd "$script_dir/../.." && pwd)"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/trained_masked_mqar-v1}")"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$OUTPUT_ROOT"
exports="ALL,NEOBERT_ROOT=$NEOBERT_ROOT,COMPLEX_ATTENTION_ROOT=$(cd "$NEOBERT_ROOT/.." && pwd),OUTPUT_ROOT=$OUTPUT_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "preflight: array 0-1, TRANSFER_SEEDS=42, MQAR_TRAIN_SMOKE=1"
    echo "pilot: array 0-1 afterok preflight, TRANSFER_SEEDS=42"
    echo "replications: array 0-3 afterok pilot, TRANSFER_SEEDS=43:44"
    echo "comparison: afterok replications"
    exit 0
fi

preflight_id="$(sbatch --parsable --array=0-1 \
    --export="$exports,TRANSFER_SEEDS=42,MQAR_TRAIN_SMOKE=1" \
    "$script_dir/trained_masked_mqar.sbatch")"
pilot_id="$(sbatch --parsable --array=0-1 --dependency="afterok:$preflight_id" \
    --export="$exports,TRANSFER_SEEDS=42,MQAR_TRAIN_SMOKE=0" \
    "$script_dir/trained_masked_mqar.sbatch")"
replication_id="$(sbatch --parsable --array=0-3 --dependency="afterok:$pilot_id" \
    --export="$exports,TRANSFER_SEEDS=43:44,MQAR_TRAIN_SMOKE=0" \
    "$script_dir/trained_masked_mqar.sbatch")"
comparison_id="$(sbatch --parsable --dependency="afterok:$replication_id" \
    --export="$exports" \
    "$script_dir/compare_trained_masked_mqar.sbatch")"

printf 'preflight_job_id=%s\n' "$preflight_id"
printf 'pilot_job_id=%s\n' "$pilot_id"
printf 'replication_job_id=%s\n' "$replication_id"
printf 'comparison_job_id=%s\n' "$comparison_id"
printf 'output_root=%s\n' "$OUTPUT_ROOT"
