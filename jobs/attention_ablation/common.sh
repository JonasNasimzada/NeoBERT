#!/usr/bin/env bash

# Shared helpers for the calibrated seven-way attention ablation.

ATTENTION_VARIANTS=(
    complex-native
    complex-torch
    complex-flash
    split-native
    split-torch
    real-torch
    real-flash
)
ATTENTION_SPACES=(complex complex complex split split real real)
ATTENTION_BACKENDS=(native torch flash native torch torch flash)
MODEL_CONFIGS=(
    attention-ablation-complex
    attention-ablation-complex
    attention-ablation-complex
    attention-ablation-split
    attention-ablation-split
    attention-ablation-real
    attention-ablation-real
)

# Equal-step/equal-token budget for the controlled seven-way sweep. At effective
# batch 32 and context 512, every model sees 688,128,000 token positions.
ATTENTION_EQUAL_TOKEN_STEPS=42000

resolve_attention_variant() {
    local task_id="${1:?array task id is required}"
    if [[ ! "$task_id" =~ ^[0-6]$ ]]; then
        echo "Array task id must be an integer from 0 through 6; got '$task_id'." >&2
        return 2
    fi
    ATTENTION_VARIANT="${ATTENTION_VARIANTS[$task_id]}"
    ATTENTION_SPACE="${ATTENTION_SPACES[$task_id]}"
    ATTENTION_BACKEND="${ATTENTION_BACKENDS[$task_id]}"
    MODEL_CONFIG="${MODEL_CONFIGS[$task_id]}"
    ATTENTION_TARGET_STEPS="$ATTENTION_EQUAL_TOKEN_STEPS"
    export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND MODEL_CONFIG
    export ATTENTION_TARGET_STEPS
}

validate_experiment_id() {
    local experiment_id="${1:?experiment id is required}"
    if [[ ! "$experiment_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "EXPERIMENT_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
        return 2
    fi
}

validate_attention_seed() {
    local seed="${1:?seed is required}"
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "SEED must be a nonnegative integer; got '$seed'." >&2
        return 2
    fi
}

setup_attention_runtime() {
    local common_dir neobert_default cache_root missing_modules
    common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    neobert_default="$(cd "$common_dir/../.." && pwd)"

    NEOBERT_ROOT="${NEOBERT_ROOT:-$neobert_default}"
    COMPLEX_ATTENTION_ROOT="${COMPLEX_ATTENTION_ROOT:-$(cd "$NEOBERT_ROOT/.." && pwd)}"
    CONDA_ENV_NAME="${CONDA_ENV_NAME:-attention_dev}"
    export NEOBERT_ROOT COMPLEX_ATTENTION_ROOT CONDA_ENV_NAME

    if ! command -v module >/dev/null 2>&1; then
        echo "The cluster module command is unavailable; cannot initialize Conda." >&2
        return 2
    fi
    module purge
    module load "${MINICONDA_MODULE:-Miniconda3}"
    module load "${CUDA_MODULE:-CUDA}"
    if [[ -z "${EBROOTMINICONDA3:-}" || ! -f "$EBROOTMINICONDA3/bin/activate" ]]; then
        echo "Miniconda3 did not provide EBROOTMINICONDA3/bin/activate." >&2
        return 2
    fi
    source "$EBROOTMINICONDA3/bin/activate"
    conda activate "$CONDA_ENV_NAME"

    COMPLEX_ATTN_PYTHON="$(command -v python)"
    export COMPLEX_ATTN_PYTHON
    if [[ ! -x "$COMPLEX_ATTN_PYTHON" ]]; then
        echo "Activated environment has no executable Python: $CONDA_ENV_NAME" >&2
        return 2
    fi

    # Match an interactive `conda activate`: this environment currently uses
    # both Conda and user-site packages for its pinned Python training stack.
    unset PYTHONNOUSERSITE
    export PYTHONPATH="$NEOBERT_ROOT/src:$COMPLEX_ATTENTION_ROOT${CUSTOM_TORCH_PYTHONPATH:+:$CUSTOM_TORCH_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"

    # The editable PyTorch install loads its extension and matching libraries
    # from attention_dev. Do not prepend pytorch/torch/lib here: that directory
    # can contain artifacts from an older source revision and can poison child
    # extension linkers even though torch._C itself has an $ORIGIN/lib RPATH.

    cache_root="${SLURM_TMPDIR:-/tmp}/complex-attention-${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
    mkdir -p "$cache_root/triton" "$cache_root/inductor" "$cache_root/cuda"
    export TRITON_CACHE_DIR="$cache_root/triton"
    export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
    export CUDA_CACHE_PATH="$cache_root/cuda"
    export HF_HOME="${HF_HOME:-$COMPLEX_ATTENTION_ROOT/.cache/huggingface}"

    missing_modules="$("$COMPLEX_ATTN_PYTHON" -c '
import importlib.util

required = (
    "typing_extensions",
    "torch",
    "hydra",
    "omegaconf",
    "datasets",
    "transformers",
    "accelerate",
    "deepspeed",
    "wandb",
)
print(" ".join(name for name in required if importlib.util.find_spec(name) is None))
')"
    if [[ -n "$missing_modules" ]]; then
        echo "Conda environment '$CONDA_ENV_NAME' is missing required modules: $missing_modules" >&2
        echo "Install the NeoBERT Python stack once with:" >&2
        echo "  PYTHONNOUSERSITE=1 python -m pip install -r $NEOBERT_ROOT/requirements-optibertneo-h100.txt" >&2
        return 2
    fi
}
