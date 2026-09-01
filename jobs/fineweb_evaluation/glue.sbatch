#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-glue
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-53%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=ca-fw-glue-%A_%a.out

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

task_id="${SLURM_ARRAY_TASK_ID:?This job must run as an array task.}"
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( 10#$task_id > 53 )); then
    echo "Array task id must be in 0..53; got $task_id" >&2
    exit 2
fi
tasks=(cola sst2 mrpc stsb qqp mnli qnli rte wnli)
seeds=(42 43 44)
seed_index=$((10#$task_id / 18))
within_seed=$((10#$task_id % 18))
model_index=$((within_seed / 9))
task_index=$((within_seed % 9))
TASK="${tasks[$task_index]}"
SEED="${seeds[$seed_index]}"

case "$model_index" in
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
        echo "Internal model index is invalid: $model_index" >&2
        exit 2
        ;;
esac
ATTENTION_BACKEND="flash"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
source "$common_file"
setup_attention_runtime

GLUE_OUTPUT_ROOT="${GLUE_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/glue-validation-v1}"
GLUE_DATA_ROOT="${GLUE_DATA_ROOT:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/glue/nyu-mll-glue-bcdcba79}"
OUTPUT_DIR="$GLUE_OUTPUT_ROOT/$ATTENTION_VARIANT/seed-$SEED/$TASK"
PAIR_PREFLIGHT="$GLUE_OUTPUT_ROOT/preflight/paired_preflight.json"
REPORT="$OUTPUT_DIR/report.json"
mkdir -p "$OUTPUT_DIR"
if [[ "${FORCE_GLUE_RERUN:-0}" != "1" && -s "$REPORT" ]]; then
    echo "Skipping completed GLUE run: $ATTENTION_VARIANT seed=$SEED task=$TASK"
    exit 0
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEOBERT_BABYLM_FINETUNE_COMPAT=1
export NEOBERT_BABYLM_SEED="$SEED"
export PYTHONPATH="$NEOBERT_ROOT/scripts/fineweb_eval/babylm_finetune_compat:$PYTHONPATH"

RESOURCE_REPORT="$OUTPUT_DIR/resource_usage.json"
"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/attention_ablation/run_with_resource_monitor.py" \
    --output "$RESOURCE_REPORT" \
    --sample-interval "${BENCHMARK_MEMORY_SAMPLE_INTERVAL:-0.25}" \
    -- "$COMPLEX_ATTN_PYTHON" \
        "$NEOBERT_ROOT/scripts/fineweb_evaluation/glue.py" \
        --model "$MODEL_PATH" \
        --expected-space "$ATTENTION_SPACE" \
        --variant "$ATTENTION_VARIANT" \
        --task "$TASK" \
        --seed "$SEED" \
        --data-root "$GLUE_DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --paired-preflight "$PAIR_PREFLIGHT" \
        --epochs "${GLUE_EPOCHS:-3}" \
        --train-batch-size "${GLUE_TRAIN_BATCH_SIZE:-32}" \
        --evaluation-batch-size "${GLUE_EVALUATION_BATCH_SIZE:-64}" \
        --learning-rate "${GLUE_LEARNING_RATE:-2e-5}" \
        --weight-decay "${GLUE_WEIGHT_DECAY:-0.01}" \
        --warmup-ratio "${GLUE_WARMUP_RATIO:-0.06}" \
        --max-grad-norm "${GLUE_MAX_GRAD_NORM:-1.0}" \
        --max-length "${GLUE_MAX_LENGTH:-128}" \
        --classifier-dropout "${GLUE_CLASSIFIER_DROPOUT:-0.1}"
