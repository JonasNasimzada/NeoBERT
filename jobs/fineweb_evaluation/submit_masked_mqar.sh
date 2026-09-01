#!/usr/bin/env bash
# Submit paired A100 masked-MQAR smoke/full evaluations and comparison.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEOBERT_ROOT="$(cd "$script_dir/../.." && pwd)"
MQAR_SEED="${MQAR_SEED:-20260831}"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/masked_mqar-seed-$MQAR_SEED}")"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$OUTPUT_ROOT/smoke" "$OUTPUT_ROOT/full"

if [[ "$SKIP_SMOKE" != "0" && "$SKIP_SMOKE" != "1" ]]; then
    echo "SKIP_SMOKE must be 0 or 1; got '$SKIP_SMOKE'." >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1; got '$DRY_RUN'." >&2
    exit 2
fi

export_values="ALL,NEOBERT_ROOT=$NEOBERT_ROOT,OUTPUT_ROOT=$OUTPUT_ROOT,MQAR_SEED=$MQAR_SEED"
if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$SKIP_SMOKE" == "0" ]]; then
        echo sbatch --parsable --export="$export_values,MQAR_SMOKE=1" "$script_dir/masked_mqar.sbatch"
        echo sbatch --parsable --dependency=afterok:SMOKE_JOB_ID --export="$export_values,MQAR_SMOKE=0" "$script_dir/masked_mqar.sbatch"
    else
        echo sbatch --parsable --export="$export_values,MQAR_SMOKE=0" "$script_dir/masked_mqar.sbatch"
    fi
    echo sbatch --parsable --dependency=afterok:FULL_JOB_ID --export="$export_values" "$script_dir/compare_masked_mqar.sbatch"
    exit 0
fi

if [[ "$SKIP_SMOKE" == "0" ]]; then
    smoke_job_id="$(
        sbatch --parsable \
            --export="$export_values,MQAR_SMOKE=1" \
            "$script_dir/masked_mqar.sbatch"
    )"
    full_dependency="--dependency=afterok:$smoke_job_id"
else
    smoke_job_id="skipped"
    full_dependency=""
fi

full_job_id="$(
    sbatch --parsable \
        ${full_dependency:+"$full_dependency"} \
        --export="$export_values,MQAR_SMOKE=0" \
        "$script_dir/masked_mqar.sbatch"
)"
comparison_job_id="$(
    sbatch --parsable \
        --dependency="afterok:$full_job_id" \
        --export="$export_values" \
        "$script_dir/compare_masked_mqar.sbatch"
)"

printf 'smoke_job_id=%s\n' "$smoke_job_id"
printf 'full_job_id=%s\n' "$full_job_id"
printf 'comparison_job_id=%s\n' "$comparison_job_id"
printf 'output_root=%s\n' "$OUTPUT_ROOT"
