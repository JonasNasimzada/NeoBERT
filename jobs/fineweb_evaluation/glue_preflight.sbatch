#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-glue-pre
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=ca-fw-glue-pre-%j.out

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/fineweb_evaluation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/fineweb_evaluation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
common_file="$job_dir/../attention_ablation/common.sh"
if [[ ! -f "$common_file" ]]; then
    echo "Cannot locate attention ablation runtime from $job_dir" >&2
    exit 2
fi

ATTENTION_VARIANT="multispace-flash"
ATTENTION_SPACE="multispace"
ATTENTION_BACKEND="flash"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
source "$common_file"
setup_attention_runtime

MULTISPACE_MODEL_PATH="${MULTISPACE_MODEL_PATH:-$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model}"
REAL_MODEL_PATH="${REAL_MODEL_PATH:-$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model}"
GLUE_OUTPUT_ROOT="${GLUE_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/glue-validation-v1}"
GLUE_DATA_ROOT="${GLUE_DATA_ROOT:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/glue/nyu-mll-glue-bcdcba79}"
PREFLIGHT_ROOT="$GLUE_OUTPUT_ROOT/preflight"
mkdir -p "$PREFLIGHT_ROOT/multispace-flash" "$PREFLIGHT_ROOT/real-flash"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEOBERT_BABYLM_FINETUNE_COMPAT=1
export NEOBERT_BABYLM_SEED=42
export PYTHONPATH="$NEOBERT_ROOT/scripts/fineweb_eval/babylm_finetune_compat:$PYTHONPATH"

"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/fineweb_evaluation/glue_prepare_data.py" \
    --output "$GLUE_DATA_ROOT" \
    --verify-only

"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/fineweb_evaluation/glue.py" \
    --model "$MULTISPACE_MODEL_PATH" \
    --expected-space multispace \
    --variant multispace-flash \
    --task cola \
    --seed 42 \
    --data-root "$GLUE_DATA_ROOT" \
    --output-dir "$PREFLIGHT_ROOT/multispace-flash" \
    --preflight

"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/fineweb_evaluation/glue.py" \
    --model "$REAL_MODEL_PATH" \
    --expected-space real \
    --variant real-flash \
    --task cola \
    --seed 42 \
    --data-root "$GLUE_DATA_ROOT" \
    --output-dir "$PREFLIGHT_ROOT/real-flash" \
    --preflight

"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/fineweb_evaluation/check_glue_preflight.py" \
    --multispace "$PREFLIGHT_ROOT/multispace-flash/preflight.json" \
    --real "$PREFLIGHT_ROOT/real-flash/preflight.json" \
    --output "$PREFLIGHT_ROOT/paired_preflight.json"
