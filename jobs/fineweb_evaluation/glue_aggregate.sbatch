#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-glue-agg
#SBATCH --partition=slowlane
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=ca-fw-glue-agg-%j.out

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/fineweb_evaluation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/fineweb_evaluation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
common_file="$job_dir/../attention_ablation/common.sh"
ATTENTION_VARIANT="real-torch"
ATTENTION_SPACE="real"
ATTENTION_BACKEND="torch"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
source "$common_file"
setup_attention_runtime

GLUE_OUTPUT_ROOT="${GLUE_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/glue-validation-v1}"
"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/fineweb_evaluation/aggregate_glue.py" \
    --root "$GLUE_OUTPUT_ROOT" \
    --output "$GLUE_OUTPUT_ROOT/paired_comparison.json"
