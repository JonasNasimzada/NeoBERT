#!/usr/bin/env bash
#SBATCH --job-name=ca-scaled-a100-test
#SBATCH --account=hk-project-pai00130
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

# HoreKa Green wrapper for the cluster-independent scaled-model preflight.
# Submit the dependency on the command line, for example:
#   sbatch --dependency=afterok:5125968 --kill-on-invalid-dep=yes \
#     jobs/scaled_fineweb/preflight-horeka-a100.sbatch

export SOURCE="${SOURCE:-/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention}"
export NEOBERT_ROOT="${NEOBERT_ROOT:-$SOURCE/NeoBERT}"
export COMPLEX_ATTENTION_ROOT="${COMPLEX_ATTENTION_ROOT:-$SOURCE}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hkfs/home/project/hk-project-pai00051/st_st171793}"
export MAMBA_ENV="${MAMBA_ENV:-$MAMBA_ROOT_PREFIX/envs/attention_dev}"
export MAMBA_EXE="${MAMBA_EXE:-$MAMBA_ROOT_PREFIX/micromamba/micromamba}"

for required_path in \
    "$NEOBERT_ROOT/jobs/scaled_fineweb/preflight.sbatch" \
    "$NEOBERT_ROOT/scripts/attention_ablation/validate_scaled_pairs.py" \
    "$NEOBERT_ROOT/scripts/attention_ablation/smoke_scaled_model.py" \
    "$MAMBA_EXE" \
    "$MAMBA_ENV/bin/python"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required HoreKa path is missing: $required_path" >&2
        exit 2
    fi
done

module purge
module load compiler/gnu/13
module load devel/cuda/12.9
module list
nvcc --version

eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate "$MAMBA_ENV"
export COMPLEX_ATTN_PYTHON="$MAMBA_ENV/bin/python"
if [[ "$(realpath -e "$(command -v python)")" != "$(realpath -e "$COMPLEX_ATTN_PYTHON")" ]]; then
    echo "micromamba activated $(command -v python), expected $COMPLEX_ATTN_PYTHON" >&2
    exit 2
fi

export PYTHONPATH="$NEOBERT_ROOT/src:$COMPLEX_ATTENTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SLURM_TMPDIR="${TMPDIR:?HoreKa did not provide its node-local TMPDIR}"
cache_root="$TMPDIR/complex-attention-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$cache_root/triton" "$cache_root/inductor" "$cache_root/cuda"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
export CUDA_CACHE_PATH="$cache_root/cuda"
export HF_HOME="${HF_HOME:-$COMPLEX_ATTENTION_ROOT/.cache/huggingface}"
export ATTENTION_RUNTIME_PRECONFIGURED=1

missing_modules="$("$COMPLEX_ATTN_PYTHON" - <<'PY'
import importlib.util

required = (
    "torch",
    "hydra",
    "omegaconf",
    "transformers",
    "complex_attention",
    "triton",
)
print(" ".join(name for name in required if importlib.util.find_spec(name) is None))
PY
)"
if [[ -n "$missing_modules" ]]; then
    echo "Micromamba environment is missing required modules: $missing_modules" >&2
    exit 2
fi

cd "$NEOBERT_ROOT"
exec bash jobs/scaled_fineweb/preflight.sbatch
