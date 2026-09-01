#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-babylm-ft
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-13%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=ca-fw-babylm-ft-%A_%a.out

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
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( 10#$task_id > 13 )); then
    echo "Array task id must be an integer from 0 through 13; got $task_id" >&2
    exit 2
fi
tasks=(boolq multirc rte wsc mrpc qqp mnli)
task_index=$((10#$task_id % ${#tasks[@]}))
model_index=$((10#$task_id / ${#tasks[@]}))
TASK="${tasks[$task_index]}"

case "$model_index" in
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
        echo "Internal model index is invalid: $model_index" >&2
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
if [[ ! -d "$BABYLM_EVAL_DEPS" ]]; then
    echo "BabyLM dependency overlay is missing: $BABYLM_EVAL_DEPS" >&2
    exit 2
fi

EVAL_ROOT="$(realpath -m "${FINEWEB_EVAL_ROOT:-$NEOBERT_ROOT/logs/fineweb_eval/babylm-2026-strict-external}")"
RESULTS_ROOT="$EVAL_ROOT/results"
FINETUNED_ROOT="$EVAL_ROOT/finetuned_models"
MODEL_LINK_ROOT="$EVAL_ROOT/models"
mkdir -p "$RESULTS_ROOT" "$FINETUNED_ROOT" "$MODEL_LINK_ROOT"
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

LR="${BABYLM_FINETUNE_LR:-3e-5}"
SEED="${SEED:-42}"
EPOCHS="${BABYLM_FINETUNE_EPOCHS:-10}"
BATCH_SIZE="${BABYLM_FINETUNE_BATCH_SIZE:-32}"
NUM_LABELS=2
METRIC_FOR_VALID="accuracy"
METRICS=(accuracy f1 mcc)
case "$TASK" in
    boolq|multirc)
        BATCH_SIZE="${BABYLM_FINETUNE_BIG_BATCH_SIZE:-16}"
        ;;
    wsc)
        EPOCHS="${BABYLM_FINETUNE_WSC_EPOCHS:-30}"
        ;;
    mrpc|qqp)
        METRIC_FOR_VALID="f1"
        ;;
    mnli)
        NUM_LABELS=3
        METRICS=(accuracy)
        ;;
esac

task_data="$BABYLM_EVAL_DATA/glue_filtered"
for split in train valid; do
    if [[ ! -s "$task_data/$TASK.$split.jsonl" ]]; then
        echo "BabyLM fine-tuning data is missing: $task_data/$TASK.$split.jsonl" >&2
        exit 2
    fi
done
result_file="$RESULTS_ROOT/$EVAL_MODEL_NAME/main/finetune/$TASK/results.txt"
predictions_file="$RESULTS_ROOT/$EVAL_MODEL_NAME/main/finetune/$TASK/predictions.json"
checkpoint="$FINETUNED_ROOT/$EVAL_MODEL_NAME/$TASK/model.pt"
manifest="$FINETUNED_ROOT/$EVAL_MODEL_NAME/$TASK/checkpoint_manifest.json"
if [[ "${BABYLM_FORCE_RERUN:-0}" != "1" \
    && -s "$result_file" && -s "$predictions_file" \
    && -s "$checkpoint" && -s "$manifest" ]]; then
    echo "Skipping completed BabyLM fine-tune task: $EVAL_MODEL_NAME/$TASK"
    exit 0
fi

export NEOBERT_BABYLM_FINETUNE_COMPAT=1
export NEOBERT_BABYLM_SEED="$SEED"
export PYTHONPATH="$NEOBERT_ROOT/scripts/fineweb_eval/babylm_finetune_compat:$BABYLM_EVAL_ROOT/strict:$BABYLM_EVAL_DEPS:$PYTHONPATH"

cd "$BABYLM_EVAL_WORKDIR"
resource_report="$EVAL_ROOT/$EVAL_MODEL_NAME-finetune-$TASK-resource-usage.json"
"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/attention_ablation/run_with_resource_monitor.py" \
    --output "$resource_report" \
    --sample-interval "${BENCHMARK_MEMORY_SAMPLE_INTERVAL:-0.25}" \
    -- "$COMPLEX_ATTN_PYTHON" -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_LINK" \
        --train_data "$task_data/$TASK.train.jsonl" \
        --valid_data "$task_data/$TASK.valid.jsonl" \
        --predict_data "$task_data/$TASK.valid.jsonl" \
        --task "$TASK" \
        --num_labels "$NUM_LABELS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LR" \
        --num_epochs "$EPOCHS" \
        --sequence_length 512 \
        --results_dir "$RESULTS_ROOT" \
        --save \
        --save_dir "$FINETUNED_ROOT" \
        --metrics "${METRICS[@]}" \
        --metric_for_valid "$METRIC_FOR_VALID" \
        --seed "$SEED" \
        --verbose \
        --padding_side left \
        --take_final

"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/fineweb_eval/write_babylm_finetune_manifest.py" \
    --task "$TASK" \
    --base-model "$MODEL_LINK" \
    --checkpoint "$checkpoint" \
    --result "$result_file" \
    --output "$manifest" \
    --learning-rate "$LR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --metric-for-valid "$METRIC_FOR_VALID"
