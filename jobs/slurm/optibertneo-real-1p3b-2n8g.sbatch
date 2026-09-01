#!/usr/bin/env bash
#SBATCH --job-name=optibertneo-real-1p3b
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --kill-on-invalid-dep=yes

set -euo pipefail

# Submit from the NeoBERT checkout:
#   cd /path/to/ComplexAttention/NeoBERT
#   sbatch --partition=<h100-partition> jobs/slurm/optibertneo-real-1p3b-2n8g.sbatch
#
# Site-specific account, partition, constraint, QoS, and reservation flags
# deliberately stay on the sbatch command line. The preflight rejects anything
# other than two nodes with four visible H100s per node.

project_root=${OPTIBERT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ ! -f "$project_root/scripts/pretraining/pretrain.py" ]]; then
    echo "OPTIBERT_PROJECT_ROOT/SLURM_SUBMIT_DIR is not a NeoBERT checkout: $project_root" >&2
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
export OPTIBERT_PROJECT_ROOT=$project_root
export PYTHON_BIN=$OPTIBERT_PYTHON
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
# Keep the sibling PyTorch source tree off PYTHONPATH/LD_LIBRARY_PATH: the
# selected OPTIBERT_PYTHON contains the separately built SM90 wheel.
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

export NUM_MACHINES=2
export GPUS_PER_NODE=4
export EXPECTED_NUM_MACHINES=2
export EXPECTED_WORLD_SIZE=8
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / GPUS_PER_NODE))

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_MODE=${WANDB_MODE:-offline}
export MAX_TIME_SECONDS=${MAX_TIME_SECONDS:-13200}
if [[ ! "$MAX_TIME_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TIME_SECONDS must be a positive integer, got: $MAX_TIME_SECONDS" >&2
    exit 2
fi
optibert_job_started_epoch=$(date +%s)
export OPTIBERT_TRAINING_DEADLINE_EPOCH=$((optibert_job_started_epoch + MAX_TIME_SECONDS))
export OPTIBERT_DATASET=${OPTIBERT_DATASET:-$project_root/tokenized_datasets/fineweb_edu_roberta_1p6b}
export RUN_ROOT=${RUN_ROOT:-$project_root/logs/optibertneo-real-1p3b/$SLURM_JOB_ID}
export HF_HOME=${HF_HOME:-$project_root/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

# Each compute node has an independent path with the same name.
cache_root=${SLURM_TMPDIR:-/tmp}/optibertneo-$SLURM_JOB_ID
export TRITON_CACHE_DIR=$cache_root/triton
export TORCHINDUCTOR_CACHE_DIR=$cache_root/inductor

echo "job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES master=$MASTER_ADDR:$MASTER_PORT"
echo "project=$project_root dataset=$OPTIBERT_DATASET run=$RUN_ROOT max_time_seconds=$MAX_TIME_SECONDS deadline_epoch=$OPTIBERT_TRAINING_DEADLINE_EPOCH"

srun \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --gpus-per-task=4 \
    --gpu-bind=none \
    --kill-on-bad-exit=1 \
    "$project_root/jobs/slurm/run-optibertneo-real-node.sh"
