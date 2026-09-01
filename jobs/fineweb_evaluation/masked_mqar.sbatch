#!/usr/bin/env bash
#SBATCH --job-name=ca-masked-mqar
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=ca-masked-mqar-%A_%a.out
#SBATCH --error=ca-masked-mqar-%A_%a.err

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

case "${SLURM_ARRAY_TASK_ID:?Submit this file as its declared array.}" in
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
    *)
        echo "Array index must be 0 (multispace) or 1 (real)." >&2
        exit 2
        ;;
esac
ATTENTION_BACKEND="flash"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
setup_attention_runtime

MODEL_PATH="$(realpath "$MODEL_PATH")"
if [[ ! -f "$MODEL_PATH/config.json" \
    || ! -f "$MODEL_PATH/model.safetensors" \
    || ! -f "$MODEL_PATH/training_summary.json" ]]; then
    echo "Completed model export is incomplete: $MODEL_PATH" >&2
    exit 2
fi

MQAR_SEED="${MQAR_SEED:-20260831}"
MQAR_SMOKE="${MQAR_SMOKE:-0}"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/masked_mqar-seed-$MQAR_SEED}")"
if [[ ! "$MQAR_SEED" =~ ^[0-9]+$ ]]; then
    echo "MQAR_SEED must be a nonnegative integer; got '$MQAR_SEED'." >&2
    exit 2
fi
if [[ "$MQAR_SMOKE" != "0" && "$MQAR_SMOKE" != "1" ]]; then
    echo "MQAR_SMOKE must be 0 or 1; got '$MQAR_SMOKE'." >&2
    exit 2
fi

if [[ "$MQAR_SMOKE" == "1" ]]; then
    EVAL_KIND="smoke"
    MQAR_ARGUMENTS=(
        --contexts 128
        --bindings 4
        --distractors 0
        --distance-fractions 1/4
        --examples-per-cell 4
        --batch-tokens 512
        --warmup-iterations 1
    )
else
    EVAL_KIND="full"
    MQAR_ARGUMENTS=(
        --contexts 128 256 512 1024
        --bindings 4 8 16
        --distractors 0 16 32
        --distance-fractions 1/4 1/2 3/4
        --examples-per-cell "${MQAR_EXAMPLES_PER_CELL:-256}"
        --batch-tokens "${MQAR_BATCH_TOKENS:-4096}"
        --warmup-iterations "${MQAR_WARMUP_ITERATIONS:-2}"
    )
fi

OUTPUT_PATH="$OUTPUT_ROOT/$EVAL_KIND/$ATTENTION_VARIANT.json"
mkdir -p "$(dirname "$OUTPUT_PATH")"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "$ATTENTION_SPACE" \
    --backend flash \
    --require-a100
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/validate_100m_pair.py

echo "masked-MQAR $EVAL_KIND evaluation"
echo "variant=$ATTENTION_VARIANT"
echo "model=$MODEL_PATH"
echo "output=$OUTPUT_PATH"
echo "seed=$MQAR_SEED"

"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/masked_mqar.py \
    --model "$MODEL_PATH" \
    --variant "$ATTENTION_VARIANT" \
    --output "$OUTPUT_PATH" \
    --seed "$MQAR_SEED" \
    "${MQAR_ARGUMENTS[@]}"
