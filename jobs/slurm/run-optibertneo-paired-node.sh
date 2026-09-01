#!/usr/bin/env bash
set -euo pipefail

project_root=${OPTIBERT_PROJECT_ROOT:?OPTIBERT_PROJECT_ROOT must be exported}
python_bin=${PYTHON_BIN:?PYTHON_BIN must be exported}
variant=${OPTIBERT_VARIANT:?OPTIBERT_VARIANT must be exported as real or multispace}

case "$variant" in
    real|multispace)
        ;;
    *)
        echo "OPTIBERT_VARIANT must be real or multispace, got: $variant" >&2
        exit 2
        ;;
esac

mkdir -p \
    "${TRITON_CACHE_DIR:?TRITON_CACHE_DIR must be exported}" \
    "${TORCHINDUCTOR_CACHE_DIR:?TORCHINDUCTOR_CACHE_DIR must be exported}" \
    "${HF_HOME:?HF_HOME must be exported}" \
    "${RUN_ROOT:?RUN_ROOT must be exported}"

required_gpu=${OPTIBERT_REQUIRED_GPU:-h100}
case "$required_gpu" in
    any) gpu_requirements=(--require-gpu) ;;
    a100|h100) gpu_requirements=("--require-$required_gpu") ;;
    *)
        echo "OPTIBERT_REQUIRED_GPU must be any, a100, or h100" >&2
        exit 2
        ;;
esac

"$python_bin" "$project_root/scripts/pretraining/preflight_optibertneo.py" \
    --variant "$variant" \
    --dataset "${OPTIBERT_DATASET:?OPTIBERT_DATASET must be exported}" \
    "${gpu_requirements[@]}" \
    --expected-nodes 2 \
    --expected-gpus-per-node 4

"$python_bin" "$project_root/scripts/pretraining/validate_optibertneo_pair.py"

exec "$project_root/jobs/optibertneo-1p3b.sh" "$variant"
