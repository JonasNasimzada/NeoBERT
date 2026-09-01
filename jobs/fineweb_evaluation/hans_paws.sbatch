#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-hans-paws
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=ca-fw-hans-paws-%j.out

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
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    exit 2
fi
source "$common_file"

EVAL_TASK="${EVAL_TASK:?Set EVAL_TASK to hans or paws}"
if [[ "$EVAL_TASK" != "hans" && "$EVAL_TASK" != "paws" ]]; then
    echo "EVAL_TASK must be hans or paws; got $EVAL_TASK" >&2
    exit 2
fi

# The process loads a multispace checkpoint, so initialize its CUDA/Triton
# runtime even though the official padded classifier protocol selects the
# equivalent PyTorch attention backend for both models.
ATTENTION_VARIANT="multispace-flash"
ATTENTION_SPACE="multispace"
ATTENTION_BACKEND="flash"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
setup_attention_runtime

BABYLM_EVAL_ROOT="$(realpath "${BABYLM_EVAL_ROOT:-/mnt/nfs/home/st171793/babylm-eval-2026}")"
BABYLM_EVAL_DEPS="${BABYLM_EVAL_DEPS:-$BABYLM_EVAL_ROOT/.eval_deps}"
BABYLM_COMMIT="6f825c291e2c4c78ad33b1935fd64d45f52642dc"
if [[ "$(git -C "$BABYLM_EVAL_ROOT" rev-parse HEAD)" != "$BABYLM_COMMIT" ]]; then
    echo "BabyLM evaluator is not pinned at $BABYLM_COMMIT" >&2
    exit 2
fi

HANS_REPO="$(realpath "${HANS_REPO:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/hans}")"
HANS_DATA="$(realpath "${HANS_DATA:-$HANS_REPO/heuristics_evaluation_set.txt}")"
PAWS_DATA="$(realpath "${PAWS_DATA:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/data/paws_labeled_final/test.parquet}")"

MULTISPACE_BASE="$(realpath "${MULTISPACE_MODEL_PATH:-$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model}")"
REAL_BASE="$(realpath "${REAL_MODEL_PATH:-$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model}")"
FINETUNED_ROOT="$(realpath -m "${FINETUNED_ROOT:-$NEOBERT_ROOT/logs/fineweb_eval/babylm-2026-strict-external/finetuned_models}")"
MULTISPACE_FINETUNED="$FINETUNED_ROOT/fineweb-edu-multispace-100m-seed-42"
REAL_FINETUNED="$FINETUNED_ROOT/fineweb-edu-real-100m-seed-42"

if [[ "${PREFLIGHT:-0}" == "1" ]]; then
    LIMIT_ARGS=(--limit "${PREFLIGHT_ROWS:-32}")
    OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/hans_paws/preflight/$EVAL_TASK}")"
    walltime_label="preflight"
else
    LIMIT_ARGS=()
    OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/hans_paws/full/$EVAL_TASK}")"
    walltime_label="full"
fi
mkdir -p "$OUTPUT_ROOT"

export NEOBERT_BABYLM_FINETUNE_COMPAT=1
export NEOBERT_BABYLM_SEED="${SEED:-42}"
export PYTHONPATH="$NEOBERT_ROOT/scripts/fineweb_eval/babylm_finetune_compat:$BABYLM_EVAL_ROOT/strict:$BABYLM_EVAL_DEPS:$PYTHONPATH"

cd "$BABYLM_EVAL_ROOT/strict"
resource_report="$OUTPUT_ROOT/${EVAL_TASK}-${walltime_label}-resource-usage.json"
"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/attention_ablation/run_with_resource_monitor.py" \
    --output "$resource_report" \
    --sample-interval "${BENCHMARK_MEMORY_SAMPLE_INTERVAL:-0.25}" \
    -- "$COMPLEX_ATTN_PYTHON" \
        "$NEOBERT_ROOT/scripts/fineweb_evaluation/hans_paws.py" \
        --task "$EVAL_TASK" \
        --hans-repo "$HANS_REPO" \
        --hans-data "$HANS_DATA" \
        --paws-data "$PAWS_DATA" \
        --multispace-base "$MULTISPACE_BASE" \
        --real-base "$REAL_BASE" \
        --multispace-mnli "$MULTISPACE_FINETUNED/mnli/model.pt" \
        --real-mnli "$REAL_FINETUNED/mnli/model.pt" \
        --multispace-qqp "$MULTISPACE_FINETUNED/qqp/model.pt" \
        --real-qqp "$REAL_FINETUNED/qqp/model.pt" \
        --output-dir "$OUTPUT_ROOT" \
        --batch-size "${EVAL_BATCH_SIZE:-64}" \
        --max-length "${EVAL_MAX_LENGTH:-512}" \
        "${LIMIT_ARGS[@]}"
