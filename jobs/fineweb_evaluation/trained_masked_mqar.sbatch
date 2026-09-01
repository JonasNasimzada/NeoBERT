#!/usr/bin/env bash
#SBATCH --job-name=ca-trained-mqar
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=ca-trained-mqar-%A_%a.out
#SBATCH --error=ca-trained-mqar-%A_%a.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/fineweb_evaluation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/fineweb_evaluation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$job_dir/../attention_ablation/common.sh"

TRANSFER_SEEDS="${TRANSFER_SEEDS:-42}"
# A colon is safe inside Slurm's comma-delimited --export list.  Continue to
# accept commas for direct/local invocations, but submission helpers use ':' so
# every requested seed reaches the batch environment intact.
seed_list="${TRANSFER_SEEDS//:/,}"
IFS=',' read -r -a seed_values <<< "$seed_list"
task_id="${SLURM_ARRAY_TASK_ID:?Submit this file as an array.}"
model_index=$((task_id % 2))
seed_index=$((task_id / 2))
if (( seed_index >= ${#seed_values[@]} )); then
    echo "Array index $task_id exceeds TRANSFER_SEEDS=$TRANSFER_SEEDS" >&2
    exit 2
fi
TRANSFER_SEED="${seed_values[$seed_index]}"
if [[ ! "$TRANSFER_SEED" =~ ^[0-9]+$ ]]; then
    echo "Transfer seeds must be nonnegative integers; got '$TRANSFER_SEED'." >&2
    exit 2
fi

case "$model_index" in
    0)
        ATTENTION_VARIANT="multispace-flash"
        ATTENTION_SPACE="multispace"
        MODEL_PATH="${MULTISPACE_MODEL:-$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model}"
        ;;
    1)
        ATTENTION_VARIANT="real-flash"
        ATTENTION_SPACE="real"
        MODEL_PATH="${REAL_MODEL:-$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model}"
        ;;
esac
ATTENTION_BACKEND="flash"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
setup_attention_runtime

MODEL_PATH="$(realpath "$MODEL_PATH")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/trained_masked_mqar-v1}")"
MQAR_TRAIN_SMOKE="${MQAR_TRAIN_SMOKE:-0}"
if [[ "$MQAR_TRAIN_SMOKE" != "0" && "$MQAR_TRAIN_SMOKE" != "1" ]]; then
    echo "MQAR_TRAIN_SMOKE must be 0 or 1." >&2
    exit 2
fi
if [[ "$MQAR_TRAIN_SMOKE" == "1" ]]; then
    RUN_ROOT="$OUTPUT_ROOT/smoke/seed-$TRANSFER_SEED/$ATTENTION_VARIANT"
    PROBE_ARGUMENTS=(
        --contexts 1024
        --bindings 4
        --distractors 0
        --distance-fractions 1/4
        --steps 2
        --warmup-steps 1
        --train-batch-tokens 2048
        --eval-batch-tokens 2048
        --validation-examples-per-cell 2
        --test-examples-per-cell 2
        --log-every 1
        --no-save-model
    )
else
    RUN_ROOT="$OUTPUT_ROOT/seed-$TRANSFER_SEED/$ATTENTION_VARIANT"
    PROBE_ARGUMENTS=(
        --contexts 128 256 512 1024
        --bindings 4 8 16
        --distractors 0 16 32
        --distance-fractions 1/4 1/2 3/4
        --steps "${MQAR_TRANSFER_STEPS:-2160}"
        --warmup-steps "${MQAR_TRANSFER_WARMUP_STEPS:-108}"
        --train-batch-tokens "${MQAR_TRANSFER_BATCH_TOKENS:-4096}"
        --eval-batch-tokens "${MQAR_EVAL_BATCH_TOKENS:-4096}"
        --validation-examples-per-cell "${MQAR_VALIDATION_EXAMPLES_PER_CELL:-64}"
        --test-examples-per-cell "${MQAR_TEST_EXAMPLES_PER_CELL:-256}"
        --learning-rate "${MQAR_LEARNING_RATE:-0.00005}"
        --minimum-learning-rate "${MQAR_MINIMUM_LEARNING_RATE:-0.000005}"
        --weight-decay "${MQAR_WEIGHT_DECAY:-0.01}"
        --max-grad-norm "${MQAR_MAX_GRAD_NORM:-1.0}"
        --log-every "${MQAR_LOG_EVERY:-108}"
    )
fi
if [[ -e "$RUN_ROOT/report.json" && "${OVERWRITE_EXISTING:-0}" != "1" ]]; then
    echo "Refusing to overwrite completed trained-MQAR report: $RUN_ROOT/report.json" >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "$ATTENTION_SPACE" --backend flash --require-a100
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/validate_100m_pair.py
if [[ "$MQAR_TRAIN_SMOKE" == "1" && "$task_id" == "0" ]]; then
    "$COMPLEX_ATTN_PYTHON" -m pytest -q tests/test_trained_masked_mqar.py
fi

echo "trained masked-MQAR transfer"
echo "variant=$ATTENTION_VARIANT transfer_seed=$TRANSFER_SEED smoke=$MQAR_TRAIN_SMOKE"
echo "source_model=$MODEL_PATH"
echo "run_root=$RUN_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/trained_masked_mqar.py \
    --model "$MODEL_PATH" \
    --variant "$ATTENTION_VARIANT" \
    --output-dir "$RUN_ROOT" \
    --transfer-seed "$TRANSFER_SEED" \
    "${PROBE_ARGUMENTS[@]}"
