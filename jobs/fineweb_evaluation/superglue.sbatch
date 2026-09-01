#!/usr/bin/env bash
#SBATCH --job-name=ca-superglue
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-47%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=ca-superglue-%A_%a.out

set -euo pipefail

NEOBERT_ROOT="${NEOBERT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the NeoBERT root}}"
source "$NEOBERT_ROOT/jobs/attention_ablation/common.sh"

tasks=(boolq cb copa multirc record rte wic wsc)
seeds=(42 43 44)
task_id="${SLURM_ARRAY_TASK_ID:?array task required}"
task_index=$(((10#$task_id / 2) % 8))
model_index=$((10#$task_id % 2))
seed_index=$((10#$task_id / 16))
task="${tasks[$task_index]}"
seed="${seeds[$seed_index]}"

case "$model_index" in
    0)
        variant=multispace
        model="$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model"
        ;;
    1)
        variant=real
        model="$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model"
        ;;
esac

export ATTENTION_VARIANT="${variant}-flash"
export ATTENTION_SPACE="$variant"
export ATTENTION_BACKEND=flash
setup_attention_runtime

BABYLM_EVAL_ROOT="$(realpath "${BABYLM_EVAL_ROOT:-/mnt/nfs/home/st171793/babylm-eval-2026}")"
BABYLM_EVAL_DEPS="${BABYLM_EVAL_DEPS:-$BABYLM_EVAL_ROOT/.eval_deps}"
BABYLM_COMMIT=6f825c291e2c4c78ad33b1935fd64d45f52642dc
if [[ "$(git -C "$BABYLM_EVAL_ROOT" rev-parse HEAD)" != "$BABYLM_COMMIT" ]]; then
    echo "BabyLM classifier source is not pinned at $BABYLM_COMMIT" >&2
    exit 2
fi

data_root="${SUPERGLUE_DATA_ROOT:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/data/superglue-3de24cf8022e}"
manifest="$NEOBERT_ROOT/scripts/fineweb_evaluation/superglue_data_manifest.json"
output_root="${SUPERGLUE_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/superglue}"
output="$output_root/full/$task/$variant/seed-$seed"
report="$output/report.json"
if [[ -s "$report" && "${FORCE:-0}" != 1 ]]; then
    echo "Skipping completed SuperGLUE run: $report"
    exit 0
fi
mkdir -p "$output"

batch_size=16
eval_batch_size=64
if [[ "$task" == record ]]; then
    batch_size=8
    eval_batch_size=32
fi

export NEOBERT_BABYLM_FINETUNE_COMPAT=1
export NEOBERT_BABYLM_SEED="$seed"
export PYTHONPATH="$NEOBERT_ROOT/scripts/fineweb_eval/babylm_finetune_compat:$BABYLM_EVAL_ROOT/strict:$BABYLM_EVAL_DEPS:$PYTHONPATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

cd "$BABYLM_EVAL_ROOT/strict"
"$COMPLEX_ATTN_PYTHON" \
    "$NEOBERT_ROOT/scripts/attention_ablation/run_with_resource_monitor.py" \
    --output "$output/resource_usage.json" \
    --sample-interval 0.25 \
    -- "$COMPLEX_ATTN_PYTHON" \
        "$NEOBERT_ROOT/scripts/fineweb_evaluation/superglue.py" \
        --task "$task" \
        --model "$model" \
        --variant "$variant" \
        --data-root "$data_root" \
        --data-manifest "$manifest" \
        --output-dir "$output" \
        --seed "$seed" \
        --batch-size "$batch_size" \
        --eval-batch-size "$eval_batch_size" \
        --max-length 512 \
        --learning-rate 3e-5 \
        --weight-decay 0.01 \
        --warmup-proportion 0.06
