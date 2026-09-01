#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-babylm-preflight
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=ca-fw-babylm-preflight-%A_%a.out

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/fineweb_eval"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/fineweb_eval"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
common_file="$job_dir/../attention_ablation/common.sh"
if [[ ! -f "$common_file" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    exit 2
fi
source "$common_file"

task_id="${SLURM_ARRAY_TASK_ID:?This job must run as an array task.}"
case "$task_id" in
    0)
        ATTENTION_VARIANT="multispace-flash"
        ATTENTION_SPACE="multispace"
        MODEL_PATH="${MULTISPACE_MODEL_PATH:-$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model}"
        ;;
    1)
        ATTENTION_VARIANT="real-flash"
        ATTENTION_SPACE="real"
        MODEL_PATH="${REAL_MODEL_PATH:-$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model}"
        ;;
    *)
        echo "Array task id must be 0 or 1; got $task_id" >&2
        exit 2
        ;;
esac
ATTENTION_BACKEND="flash"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
setup_attention_runtime

BABYLM_EVAL_ROOT="$(realpath "${BABYLM_EVAL_ROOT:-/mnt/nfs/home/st171793/babylm-eval-2026}")"
BABYLM_EVAL_DEPS="${BABYLM_EVAL_DEPS:-$BABYLM_EVAL_ROOT/.eval_deps}"
PINNED_COMMIT="6f825c291e2c4c78ad33b1935fd64d45f52642dc"
if [[ "$(git -C "$BABYLM_EVAL_ROOT" rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
    echo "BabyLM evaluator is not at pinned commit $PINNED_COMMIT" >&2
    exit 2
fi
if [[ ! -d "$BABYLM_EVAL_DEPS" ]]; then
    echo "BabyLM dependency overlay is missing: $BABYLM_EVAL_DEPS" >&2
    exit 2
fi

export NEOBERT_BABYLM_FORCE_BF16=1
export NEOBERT_BABYLM_SEED="${SEED:-42}"
export PYTHONPATH="$NEOBERT_ROOT/scripts/attention_ablation/babylm_compat:$BABYLM_EVAL_DEPS:$PYTHONPATH"

"$COMPLEX_ATTN_PYTHON" "$NEOBERT_ROOT/scripts/fineweb_eval/preflight_babylm_2026.py" \
    --model "$MODEL_PATH" \
    --expected-space "$ATTENTION_SPACE"
