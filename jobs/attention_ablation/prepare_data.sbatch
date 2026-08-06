#!/usr/bin/env bash
#SBATCH --job-name=ca-babylm-prep
#SBATCH --partition=slowlane
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=ca-babylm-prep-%j.out
#SBATCH --error=ca-babylm-prep-%j.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/attention_ablation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/attention_ablation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "$job_dir/common.sh" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    echo "Submit from the NeoBERT root or export NEOBERT_ROOT explicitly." >&2
    exit 2
fi
source "$job_dir/common.sh"
setup_attention_runtime

DATASET_PATH="$(realpath -m "${DATASET_PATH:?Set DATASET_PATH to a new output directory.}")"
cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/pretraining/preprocess.py \
    dataset=babylm_2026_strict \
    tokenizer=google-512 \
    dataset.path_to_disk="$DATASET_PATH"
