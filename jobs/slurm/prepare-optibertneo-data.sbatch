#!/usr/bin/env bash
#SBATCH --job-name=prepare-optibertneo-data
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00

set -euo pipefail

# This job needs outbound access to huggingface.co. Choose a CPU/data-transfer
# partition on the sbatch command line when the site provides one.

project_root=${OPTIBERT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ ! -f "$project_root/scripts/pretraining/preprocess.py" ]]; then
    echo "Submit this job from the NeoBERT repository root or export OPTIBERT_PROJECT_ROOT." >&2
    exit 1
fi
project_root=$(cd "$project_root" && pwd)
complex_root=$(cd "$project_root/.." && pwd)

if [[ -z "${OPTIBERT_PYTHON:-}" ]]; then
    echo "OPTIBERT_PYTHON must name the executable in the prepared H100 environment." >&2
    exit 1
fi
if [[ ! -x "$OPTIBERT_PYTHON" ]]; then
    echo "OPTIBERT_PYTHON is not executable: $OPTIBERT_PYTHON" >&2
    exit 1
fi
python_bin=$OPTIBERT_PYTHON
dataset_path=${OPTIBERT_DATASET:-$project_root/tokenized_datasets/fineweb_edu_roberta_1p6b}

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$complex_root:$project_root/src"
source_torch_library=$complex_root/pytorch/torch/lib
filtered_ld_library_path=
IFS=: read -r -a optibert_ld_entries <<<"${LD_LIBRARY_PATH:-}"
for optibert_ld_entry in "${optibert_ld_entries[@]}"; do
    if [[ -z "$optibert_ld_entry" || "$optibert_ld_entry" == "$source_torch_library" ]]; then
        continue
    fi
    filtered_ld_library_path+="${filtered_ld_library_path:+:}$optibert_ld_entry"
done
if [[ -n "$filtered_ld_library_path" ]]; then
    export LD_LIBRARY_PATH=$filtered_ld_library_path
else
    unset LD_LIBRARY_PATH
fi
export HF_HOME=${HF_HOME:-$project_root/.cache/huggingface}

mkdir -p "$HF_HOME" "$(dirname "$dataset_path")"

if [[ -f "$dataset_path/optibertneo_manifest.json" ]]; then
    echo "Prepared dataset already exists; validating $dataset_path"
    exec "$python_bin" "$project_root/scripts/pretraining/preflight_optibertneo.py" \
        --dataset "$dataset_path"
fi

"$python_bin" "$project_root/scripts/pretraining/preprocess.py" \
    dataset=fineweb_edu \
    tokenizer=roberta \
    dataset.path_to_disk="$dataset_path"

"$python_bin" "$project_root/scripts/pretraining/preflight_optibertneo.py" \
    --dataset "$dataset_path"
