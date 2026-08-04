#!/usr/bin/env bash
set -euo pipefail

project_root=${OPTIBERT_PROJECT_ROOT:?OPTIBERT_PROJECT_ROOT must be exported}
python_bin=${PYTHON_BIN:?PYTHON_BIN must be exported}

mkdir -p \
    "${TRITON_CACHE_DIR:?TRITON_CACHE_DIR must be exported}" \
    "${TORCHINDUCTOR_CACHE_DIR:?TORCHINDUCTOR_CACHE_DIR must be exported}" \
    "${HF_HOME:?HF_HOME must be exported}" \
    "${RUN_ROOT:?RUN_ROOT must be exported}"

"$python_bin" "$project_root/scripts/pretraining/preflight_optibertneo.py" \
    --dataset "${OPTIBERT_DATASET:?OPTIBERT_DATASET must be exported}" \
    --require-gpu \
    --require-h100 \
    --expected-nodes 2 \
    --expected-gpus-per-node 4

exec "$project_root/jobs/optibertneo-1p3b.sh" real
