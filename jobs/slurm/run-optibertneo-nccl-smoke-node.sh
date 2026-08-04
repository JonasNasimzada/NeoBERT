#!/usr/bin/env bash
set -euo pipefail

project_root=${OPTIBERT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}
python_bin=${PYTHON_BIN:?PYTHON_BIN is required}

"$python_bin" "$project_root/scripts/pretraining/preflight_optibertneo.py" \
    --require-gpu \
    --require-h100 \
    --expected-nodes 2 \
    --expected-gpus-per-node 4

exec "$python_bin" -m torch.distributed.run \
    --nnodes=2 \
    --nproc-per-node=4 \
    --node-rank="${SLURM_NODEID:?SLURM_NODEID is required}" \
    --master-addr="${MASTER_ADDR:?MASTER_ADDR is required}" \
    --master-port="${MASTER_PORT:?MASTER_PORT is required}" \
    "$project_root/scripts/pretraining/distributed_smoke.py"
