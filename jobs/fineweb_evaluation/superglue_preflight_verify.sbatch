#!/usr/bin/env bash
#SBATCH --job-name=ca-superglue-gate
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=ca-superglue-gate-%j.out

set -euo pipefail
NEOBERT_ROOT="${NEOBERT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the NeoBERT root}}"
source "$NEOBERT_ROOT/jobs/attention_ablation/common.sh"
export ATTENTION_SPACE=real ATTENTION_BACKEND=flash ATTENTION_VARIANT=real-flash
setup_attention_runtime

output_root="${SUPERGLUE_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/superglue}"
"$COMPLEX_ATTN_PYTHON" "$NEOBERT_ROOT/scripts/fineweb_evaluation/verify_superglue_preflight.py" \
    --input-root "$output_root/preflight" \
    --output "$output_root/preflight/fairness_gate.json" \
    --seed 42
