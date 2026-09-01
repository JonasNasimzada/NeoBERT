#!/usr/bin/env bash
#SBATCH --job-name=multispace-fineweb-prep
#SBATCH --partition=slowlane
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=multispace-fineweb-prep-%j.out
#SBATCH --error=multispace-fineweb-prep-%j.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    common_path="$NEOBERT_ROOT/jobs/attention_ablation/common.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" \
    && -f "$SLURM_SUBMIT_DIR/jobs/attention_ablation/common.sh" ]]; then
    common_path="$SLURM_SUBMIT_DIR/jobs/attention_ablation/common.sh"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    common_path="$script_dir/../attention_ablation/common.sh"
fi
if [[ ! -f "$common_path" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh at $common_path" >&2
    echo "Submit from the NeoBERT root or export NEOBERT_ROOT explicitly." >&2
    exit 2
fi
source "$common_path"
setup_attention_runtime

DATASET_PATH="$(realpath -m "${DATASET_PATH:-$NEOBERT_ROOT/tokenized_datasets/fineweb_edu_google_1024_1p6b}")"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/pretraining/preprocess.py \
    dataset=fineweb_edu_google_1024 \
    tokenizer=google-1024 \
    dataset.path_to_disk="$DATASET_PATH"
