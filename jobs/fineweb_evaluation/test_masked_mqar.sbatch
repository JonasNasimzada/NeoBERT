#!/usr/bin/env bash
#SBATCH --job-name=ca-mqar-tests
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=ca-mqar-tests-%j.out
#SBATCH --error=ca-mqar-tests-%j.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/fineweb_evaluation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/fineweb_evaluation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$job_dir/../attention_ablation/common.sh"
ATTENTION_SPACE="multispace"
ATTENTION_BACKEND="flash"
export ATTENTION_SPACE ATTENTION_BACKEND
setup_attention_runtime

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space multispace \
    --backend flash \
    --require-a100
"$COMPLEX_ATTN_PYTHON" -m pytest -q tests/test_masked_mqar.py
