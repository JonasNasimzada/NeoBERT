#!/usr/bin/env bash
#SBATCH --job-name=ca-babylm-train
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=15:00:00
#SBATCH --output=ca-babylm-train-%A_%a.out

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
EXPERIMENT_ID="${EXPERIMENT_ID:-a100-s1024-1p376b-v1}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-$ATTENTION_TARGET_STEPS}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-53640}"
validate_experiment_id "$EXPERIMENT_ID"
validate_attention_seed "$SEED"
if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer; got '$MAX_STEPS'." >&2
    exit 2
fi
WARMUP_STEPS="${WARMUP_STEPS:-$(( (MAX_STEPS * 100 + 1535) / 1536 ))}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-$(( (MAX_STEPS + 5) / 6 ))}"
EVAL_STEPS="${EVAL_STEPS:-$CHECKPOINT_STEPS}"
LOG_INTERVAL="${LOG_INTERVAL:-$(( (MAX_STEPS + 99) / 100 ))}"
for schedule_name in WARMUP_STEPS CHECKPOINT_STEPS EVAL_STEPS LOG_INTERVAL; do
    schedule_value="${!schedule_name}"
    if [[ ! "$schedule_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$schedule_name must be a positive integer; got '$schedule_value'." >&2
        exit 2
    fi
done
RUNS_ROOT="$RUNS_ROOT/$EXPERIMENT_ID"
RUN_ROOT="$RUNS_ROOT/$ATTENTION_VARIANT/seed-$SEED"
WANDB_RUN_ID="ca-${EXPERIMENT_ID}-${ATTENTION_VARIANT}-s${SEED}"

if (( MICRO_BATCH * GRAD_ACCUM != ATTENTION_TARGET_SEQUENCE_BATCH )); then
    echo "MICRO_BATCH * GRAD_ACCUM must remain $ATTENTION_TARGET_SEQUENCE_BATCH for the controlled sweep." >&2
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
    dataset.pack_to_length="$ATTENTION_TARGET_SEQUENCE_LENGTH" \
    tokenizer.max_length="$ATTENTION_TARGET_SEQUENCE_LENGTH" \
    trainer.dir="$RUN_ROOT" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.max_time_seconds="$MAX_TIME_SECONDS" \
    trainer.gradient_accumulation_steps="$GRAD_ACCUM" \
    trainer.validation.eval_steps="$EVAL_STEPS" \
    trainer.accelerate.save_steps="$CHECKPOINT_STEPS" \
    trainer.model.save_steps="$CHECKPOINT_STEPS" \
    scheduler.warmup_steps="$WARMUP_STEPS" \
    scheduler.decay_steps="$MAX_STEPS" \
    dataloader.train.batch_size="$MICRO_BATCH" \
    dataloader.train.max_length="$ATTENTION_TARGET_SEQUENCE_LENGTH" \
    dataloader.validation.batch_size="$MICRO_BATCH" \
    dataloader.validation.max_length="$ATTENTION_TARGET_SEQUENCE_LENGTH" \
    seed="$SEED" \
    wandb.log_interval="$LOG_INTERVAL" \
    wandb.id="$WANDB_RUN_ID" \
    wandb.name="$EXPERIMENT_ID-$ATTENTION_VARIANT-seed-$SEED" \
    wandb.group="complex-attention-babylm-2026-$EXPERIMENT_ID-seed-$SEED" \
    wandb.entity="${WANDB_ENTITY:-hyper_attention}" \
    wandb.project="${WANDB_PROJECT:-complex-attention-ablation}" \
    wandb.mode="${WANDB_MODE:-online}"
