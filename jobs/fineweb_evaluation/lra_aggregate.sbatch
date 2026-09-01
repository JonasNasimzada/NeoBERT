#!/usr/bin/env bash
#SBATCH --job-name=ca-lra-aggregate
#SBATCH --partition=slowlane
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=ca-lra-aggregate-%j.out
#SBATCH --error=ca-lra-aggregate-%j.err

set -euo pipefail

NEOBERT_ROOT="${NEOBERT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the NeoBERT repository root}}"
source "$NEOBERT_ROOT/jobs/attention_ablation/common.sh"
export ATTENTION_SPACE=multispace
export ATTENTION_BACKEND=flash
setup_attention_runtime

output_root="${LRA_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/lra_external}"
cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/fineweb_evaluation/lra_aggregate.py \
    --input-root "$output_root" \
    --protocol "$NEOBERT_ROOT/conf/evaluation/lra_fineweb_external.json" \
    --output "$output_root/paired_comparison.json"
