#!/usr/bin/env bash

# Shared, source-only helpers for the seven-way attention ablation.

ATTENTION_VARIANTS=(
    complex-native
    complex-torch
    complex-flash
    split-native
    split-torch
    dual-native
    dual-torch
)
ATTENTION_SPACES=(complex complex complex split split dual dual)
ATTENTION_BACKENDS=(native torch flash native torch native torch)
MODEL_CONFIGS=(
    attention-ablation-complex
    attention-ablation-complex
    attention-ablation-complex
    attention-ablation-split
    attention-ablation-split
    attention-ablation-dual
    attention-ablation-dual
)

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
    export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND MODEL_CONFIG
}

setup_attention_runtime() {
    local common_dir neobert_default cache_root
    common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    neobert_default="$(cd "$common_dir/../.." && pwd)"

    NEOBERT_ROOT="${NEOBERT_ROOT:-$neobert_default}"
    COMPLEX_ATTENTION_ROOT="${COMPLEX_ATTENTION_ROOT:-$(cd "$NEOBERT_ROOT/.." && pwd)}"
    COMPLEX_ATTN_PYTHON="${COMPLEX_ATTN_PYTHON:-/mnt/nfs/home/st171793/.conda/envs/pytorch_dev/bin/python}"
    export NEOBERT_ROOT COMPLEX_ATTENTION_ROOT COMPLEX_ATTN_PYTHON

    if [[ ! -x "$COMPLEX_ATTN_PYTHON" ]]; then
        echo "COMPLEX_ATTN_PYTHON is not executable: $COMPLEX_ATTN_PYTHON" >&2
        return 2
    fi
    export PATH="$(dirname "$COMPLEX_ATTN_PYTHON"):$PATH"

    if command -v module >/dev/null 2>&1; then
        module load "${CUDA_MODULE:-CUDA/12.6.0}"
    fi

    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$NEOBERT_ROOT/src:$COMPLEX_ATTENTION_ROOT${CUSTOM_TORCH_PYTHONPATH:+:$CUSTOM_TORCH_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
    if [[ -d "$COMPLEX_ATTENTION_ROOT/pytorch/torch/lib" ]]; then
        export LD_LIBRARY_PATH="$COMPLEX_ATTENTION_ROOT/pytorch/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    cache_root="${SLURM_TMPDIR:-/tmp}/complex-attention-${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
    mkdir -p "$cache_root/triton" "$cache_root/inductor" "$cache_root/cuda"
    export TRITON_CACHE_DIR="$cache_root/triton"
    export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
    export CUDA_CACHE_PATH="$cache_root/cuda"
    export HF_HOME="${HF_HOME:-$COMPLEX_ATTENTION_ROOT/.cache/huggingface}"
}
