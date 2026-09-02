#!/usr/bin/env bash
# Prepare the pinned Google-tokenizer FineWeb-Edu DatasetDict on HoreKa.
# This is a CPU/data job; the A100 is reserved for the subsequent preflight and
# training chains.
#
# The target directory is intentionally never overwritten.  If a previous
# preparation was interrupted, choose a new output directory (or remove the
# incomplete directory only after checking that it is disposable).
#
#SBATCH --job-name=ca-fw-prep
#SBATCH --account=hk-project-pai00130
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SOURCE="${SOURCE:-/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention}"
NEOBERT_ROOT="${NEOBERT_ROOT:-$SOURCE/NeoBERT}"
COMPLEX_ATTENTION_ROOT="${COMPLEX_ATTENTION_ROOT:-$SOURCE}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hkfs/home/project/hk-project-pai00051/st_st171793}"
MAMBA_ENV="${MAMBA_ENV:-$MAMBA_ROOT_PREFIX/envs/attention_dev}"
MAMBA_EXE="${MAMBA_EXE:-$MAMBA_ROOT_PREFIX/micromamba/micromamba}"
DATASET_PATH="$(realpath -m "${DATASET_PATH:-$NEOBERT_ROOT/tokenized_datasets/fineweb_edu_google_1024_1p6b}")"

for required_path in \
    "$NEOBERT_ROOT/scripts/pretraining/preprocess.py" \
    "$MAMBA_EXE" \
    "$MAMBA_ENV/bin/python"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required HoreKa path is missing: $required_path" >&2
        exit 2
    fi
done

if [[ -e "$DATASET_PATH" ]]; then
    if [[ -f "$DATASET_PATH/dataset_dict.json" \
        && -f "$DATASET_PATH/optibertneo_manifest.json" ]]; then
        echo "Prepared dataset already exists: $DATASET_PATH"
        exit 0
    fi
    echo "Refusing to write into incomplete existing directory: $DATASET_PATH" >&2
    echo "Choose a new DATASET_PATH, or inspect and remove this directory manually." >&2
    exit 2
fi

module purge
module load compiler/gnu/13
module load devel/cuda/12.9
module list
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate "$MAMBA_ENV"
export COMPLEX_ATTN_PYTHON="$MAMBA_ENV/bin/python"
if [[ "$(realpath -e "$(command -v python)")" != "$(realpath -e "$COMPLEX_ATTN_PYTHON")" ]]; then
    echo "micromamba activated $(command -v python), expected $COMPLEX_ATTN_PYTHON" >&2
    exit 2
fi

export PYTHONPATH="$NEOBERT_ROOT/src:$COMPLEX_ATTENTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$COMPLEX_ATTENTION_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME" "$(dirname "$DATASET_PATH")"

echo "Preparing FineWeb-Edu sample-10BT"
echo "dataset_path=$DATASET_PATH"
echo "hf_home=$HF_HOME"
cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/pretraining/preprocess.py \
    dataset=fineweb_edu_google_1024 \
    tokenizer=google-1024 \
    dataset.path_to_disk="$DATASET_PATH"

if [[ ! -f "$DATASET_PATH/dataset_dict.json" \
    || ! -f "$DATASET_PATH/optibertneo_manifest.json" ]]; then
    echo "Preprocessing finished without the required DatasetDict manifest." >&2
    exit 2
fi
echo "Prepared dataset is complete: $DATASET_PATH"
