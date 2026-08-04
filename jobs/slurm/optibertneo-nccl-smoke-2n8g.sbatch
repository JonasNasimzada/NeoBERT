#!/usr/bin/env bash
#SBATCH --job-name=optibertneo-nccl-smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail

project_root=${OPTIBERT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ ! -f "$project_root/scripts/pretraining/distributed_smoke.py" ]]; then
    echo "Submit from the NeoBERT repository root or export OPTIBERT_PROJECT_ROOT." >&2
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
export EXPECTED_WORLD_SIZE=8
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
export MASTER_PORT=$((40000 + SLURM_JOB_ID % 20000))
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --gpus-per-task=4 \
    --gpu-bind=none \
    --kill-on-bad-exit=1 \
    "$project_root/jobs/slurm/run-optibertneo-nccl-smoke-node.sh"
