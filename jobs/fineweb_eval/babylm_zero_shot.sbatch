#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-babylm-zero
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=ca-fw-babylm-zero-%A_%a.out

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
        EVAL_MODEL_NAME="fineweb-edu-multispace-100m-seed-42"
        ;;
    1)
        ATTENTION_VARIANT="real-flash"
        ATTENTION_SPACE="real"
        MODEL_PATH="${REAL_MODEL_PATH:-$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model}"
        EVAL_MODEL_NAME="fineweb-edu-real-100m-seed-42"
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
BABYLM_EVAL_WORKDIR="$BABYLM_EVAL_ROOT/strict"
BABYLM_EVAL_DATA="$(realpath "${BABYLM_EVAL_DATA:-$BABYLM_EVAL_WORKDIR/evaluation_data/full_eval}")"
BABYLM_EVAL_DEPS="${BABYLM_EVAL_DEPS:-$BABYLM_EVAL_ROOT/.eval_deps}"
PINNED_COMMIT="6f825c291e2c4c78ad33b1935fd64d45f52642dc"
if [[ "$(git -C "$BABYLM_EVAL_ROOT" rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
    echo "BabyLM evaluator is not at pinned commit $PINNED_COMMIT" >&2
    exit 2
fi
if ! git -C "$BABYLM_EVAL_ROOT" diff --quiet -- \
    ':!strict/evaluation_data' ':!strict/results'; then
    echo "Pinned BabyLM evaluator has tracked source modifications." >&2
    exit 2
fi
if [[ ! -d "$BABYLM_EVAL_DEPS" ]]; then
    echo "BabyLM dependency overlay is missing: $BABYLM_EVAL_DEPS" >&2
    exit 2
fi

EVAL_ROOT="$(realpath -m "${FINEWEB_EVAL_ROOT:-$NEOBERT_ROOT/logs/fineweb_eval/babylm-2026-strict-external}")"
RESULTS_ROOT="$EVAL_ROOT/results"
MODEL_LINK_ROOT="$EVAL_ROOT/models"
mkdir -p "$RESULTS_ROOT" "$MODEL_LINK_ROOT"
MODEL_LINK="$MODEL_LINK_ROOT/$EVAL_MODEL_NAME"
if [[ -e "$MODEL_LINK" && ! -L "$MODEL_LINK" ]]; then
    echo "Refusing to replace non-symlink evaluation model path: $MODEL_LINK" >&2
    exit 2
fi
if [[ -L "$MODEL_LINK" && "$(realpath "$MODEL_LINK")" != "$(realpath "$MODEL_PATH")" ]]; then
    echo "Evaluation model symlink points at a different checkpoint: $MODEL_LINK" >&2
    exit 2
fi
if [[ ! -L "$MODEL_LINK" ]]; then
    ln -s "$(realpath "$MODEL_PATH")" "$MODEL_LINK"
fi

export NEOBERT_BABYLM_FORCE_BF16=1
export NEOBERT_BABYLM_SEED="${SEED:-42}"
export NLTK_DATA="${NLTK_DATA:-$BABYLM_EVAL_ROOT/.nltk_data}"
export PYTHONPATH="$NEOBERT_ROOT/scripts/attention_ablation/babylm_compat:$BABYLM_EVAL_DEPS:$PYTHONPATH"

cd "$BABYLM_EVAL_WORKDIR"
"$COMPLEX_ATTN_PYTHON" "$NEOBERT_ROOT/scripts/fineweb_eval/preflight_babylm_2026.py" \
    --model "$MODEL_LINK" \
    --expected-space "$ATTENTION_SPACE"

resource_report="$EVAL_ROOT/$EVAL_MODEL_NAME-zero-shot-resource-usage.json"
"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/attention_ablation/run_with_resource_monitor.py" \
    --output "$resource_report" \
    --sample-interval "${BENCHMARK_MEMORY_SAMPLE_INTERVAL:-0.25}" \
    -- bash "$NEOBERT_ROOT/scripts/fineweb_eval/run_babylm_zero_shot.sh" \
        "$MODEL_LINK" "$BABYLM_EVAL_DATA" "$RESULTS_ROOT"
