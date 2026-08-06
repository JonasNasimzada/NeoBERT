#!/usr/bin/env bash
#SBATCH --job-name=ca-babylm-train
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=ca-babylm-train-%A_%a.out
#SBATCH --error=ca-babylm-train-%A_%a.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/attention_ablation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/attention_ablation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "$job_dir/common.sh" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    echo "Submit from the NeoBERT root or export NEOBERT_ROOT explicitly." >&2
    exit 2
fi
source "$job_dir/common.sh"
resolve_attention_variant "${SLURM_ARRAY_TASK_ID:?This job must run as an array task.}"
setup_attention_runtime

DATASET_PATH="$(realpath "${DATASET_PATH:?Set DATASET_PATH to the prepared BabyLM DatasetDict directory.}")"
RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$NEOBERT_ROOT/logs/attention_ablation}")"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1536}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-9900}"
RUN_ROOT="$RUNS_ROOT/$ATTENTION_VARIANT/seed-$SEED"
WANDB_RUN_ID="ca-${ATTENTION_VARIANT}-s${SEED}"

if (( MICRO_BATCH * GRAD_ACCUM != 32 )); then
    echo "MICRO_BATCH * GRAD_ACCUM must remain 32 for the controlled sweep." >&2
    exit 2
fi

mkdir -p "$RUN_ROOT" "${WANDB_DIR:-$RUNS_ROOT/wandb}"
export WANDB_DIR="${WANDB_DIR:-$RUNS_ROOT/wandb}"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/validate_variants.py
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "$ATTENTION_SPACE" \
    --backend "$ATTENTION_BACKEND" \
    --require-a100

"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/train.py \
    model="$MODEL_CONFIG" \
    model.attention_backend="$ATTENTION_BACKEND" \
    dataset.path_to_disk="$DATASET_PATH" \
    trainer.dir="$RUN_ROOT" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.max_time_seconds="$MAX_TIME_SECONDS" \
    trainer.gradient_accumulation_steps="$GRAD_ACCUM" \
    scheduler.warmup_steps="$WARMUP_STEPS" \
    scheduler.decay_steps="$MAX_STEPS" \
    dataloader.train.batch_size="$MICRO_BATCH" \
    dataloader.validation.batch_size="$MICRO_BATCH" \
    seed="$SEED" \
    wandb.id="$WANDB_RUN_ID" \
    wandb.name="$ATTENTION_VARIANT-seed-$SEED" \
    wandb.group="complex-attention-babylm-2026-seed-$SEED" \
    wandb.entity="${WANDB_ENTITY:-null}" \
    wandb.project="${WANDB_PROJECT:-complex-attention-ablation}" \
    wandb.mode="${WANDB_MODE:-online}"
