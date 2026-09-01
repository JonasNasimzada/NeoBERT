#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-full-mlm
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=ca-fw-full-mlm-%A_%a.out
#SBATCH --error=ca-fw-full-mlm-%A_%a.err

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

task_id="${SLURM_ARRAY_TASK_ID:?This job must run as array task 0 or 1.}"
case "$task_id" in
    0)
        ATTENTION_VARIANT="multispace-flash"
        ATTENTION_SPACE="multispace"
        ATTENTION_BACKEND="flash"
        default_model_path="$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model"
        ;;
    1)
        ATTENTION_VARIANT="real-100m-flash"
        ATTENTION_SPACE="real"
        ATTENTION_BACKEND="flash"
        default_model_path="$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model"
        ;;
    *)
        echo "SLURM_ARRAY_TASK_ID must be 0 (multispace) or 1 (real-100m); got '$task_id'." >&2
        exit 2
        ;;
esac
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
setup_attention_runtime

readonly FULL_VALIDATION_TOKENS=15224832
readonly BATCH_TOKENS=4096
readonly MASK_PROBABILITY=0.20
DATASET_PATH="$(realpath "${DATASET_PATH:-$NEOBERT_ROOT/tokenized_datasets/fineweb_edu_google_1024_1p6b}")"
MODEL_PATH="$(realpath "${MODEL_PATH:-$default_model_path}")"
EVAL_ROOT="$(realpath -m "${EVAL_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/full-heldout-mlm-seed-42}")"
SEED="${SEED:-42}"
validate_attention_seed "$SEED"

for required_file in \
    "$DATASET_PATH/dataset_dict.json" \
    "$DATASET_PATH/optibertneo_manifest.json" \
    "$MODEL_PATH/config.json" \
    "$MODEL_PATH/training_summary.json"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required evaluation artifact is missing: $required_file" >&2
        exit 2
    fi
done

manifest_validation_tokens="$(
    "$COMPLEX_ATTN_PYTHON" -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["splits"]["validation"]["tokens"])' \
        "$DATASET_PATH/optibertneo_manifest.json"
)"
if [[ "$manifest_validation_tokens" != "$FULL_VALIDATION_TOKENS" ]]; then
    echo "Validation manifest contains $manifest_validation_tokens positions; expected $FULL_VALIDATION_TOKENS." >&2
    exit 2
fi

OUTPUT_DIR="$EVAL_ROOT/$ATTENTION_VARIANT"
OUTPUT_PATH="$OUTPUT_DIR/heldout_mlm_full.json"
mkdir -p "$OUTPUT_DIR" "${WANDB_DIR:-$EVAL_ROOT/wandb}"
export WANDB_DIR="${WANDB_DIR:-$EVAL_ROOT/wandb}"

echo "FineWeb-Edu full held-out MLM evaluation"
echo "variant=$ATTENTION_VARIANT"
echo "model=$MODEL_PATH"
echo "dataset=$DATASET_PATH"
echo "contexts=128,256,512,1024"
echo "validation_tokens_per_context=$FULL_VALIDATION_TOKENS"
echo "total_model_token_evaluations=$((4 * FULL_VALIDATION_TOKENS))"
echo "batch_tokens=$BATCH_TOKENS seed=$SEED mask_probability=$MASK_PROBABILITY"
echo "output=$OUTPUT_PATH"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "$ATTENTION_SPACE" \
    --backend "$ATTENTION_BACKEND" \
    --require-a100
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/benchmark_mlm.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_PATH" \
    --variant "$ATTENTION_VARIANT" \
    --contexts 128 256 512 1024 \
    --token-budget "$FULL_VALIDATION_TOKENS" \
    --batch-tokens "$BATCH_TOKENS" \
    --mask-probability "$MASK_PROBABILITY" \
    --seed "$SEED" \
    --id-prefix fineweb-edu-full-heldout-mlm \
    --project "${WANDB_PROJECT:-complex-attention-fineweb}" \
    --entity "${WANDB_ENTITY:-hyper_attention}" \
    --group fineweb-edu-full-heldout-mlm-seed-42 \
    --name "$ATTENTION_VARIANT-full-heldout-mlm" \
    --mode "${WANDB_MODE:-offline}"
