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

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$job_dir/common.sh"
setup_attention_runtime

DATASET_PATH="$(realpath -m "${DATASET_PATH:?Set DATASET_PATH to a new output directory.}")"
cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/pretraining/preprocess.py \
    dataset=babylm_2026_strict \
    tokenizer=google-512 \
    dataset.path_to_disk="$DATASET_PATH"
