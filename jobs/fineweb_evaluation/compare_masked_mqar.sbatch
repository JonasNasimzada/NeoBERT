#!/usr/bin/env bash
#SBATCH --job-name=ca-mqar-compare
#SBATCH --partition=slowlane
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=ca-mqar-compare-%j.out
#SBATCH --error=ca-mqar-compare-%j.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    NEOBERT_ROOT="$SLURM_SUBMIT_DIR"
else
    NEOBERT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
NEOBERT_ROOT="$(realpath "$NEOBERT_ROOT")"
MQAR_SEED="${MQAR_SEED:-20260831}"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/masked_mqar-seed-$MQAR_SEED}")"
MULTISPACE_REPORT="$OUTPUT_ROOT/full/multispace-flash.json"
REAL_REPORT="$OUTPUT_ROOT/full/real-flash.json"
COMPARISON_REPORT="$OUTPUT_ROOT/full/paired_comparison.json"
for report in "$MULTISPACE_REPORT" "$REAL_REPORT"; do
    if [[ ! -f "$report" ]]; then
        echo "Required paired report is missing: $report" >&2
        exit 2
    fi
done

cd "$NEOBERT_ROOT"
"${PYTHON_BIN:-python3}" scripts/attention_ablation/compare_masked_mqar.py \
    --multispace "$MULTISPACE_REPORT" \
    --real "$REAL_REPORT" \
    --output "$COMPARISON_REPORT"
