#!/usr/bin/env bash
#SBATCH --job-name=ca-fineweb-real-train
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=15:00:00
#SBATCH --output=ca-fineweb-real-train-%j.out

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/real_fineweb"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/real_fineweb"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
common_file="$job_dir/../attention_ablation/common.sh"
if [[ ! -f "$common_file" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    echo "Submit from the NeoBERT root or export NEOBERT_ROOT explicitly." >&2
    exit 2
fi
source "$common_file"

# This continuation launcher is deliberately fixed to the parameter-matched
# real FlashAttention baseline rather than an attention-ablation array task.
ATTENTION_VARIANT="real-flash"
ATTENTION_SPACE="real"
ATTENTION_BACKEND="flash"
MODEL_CONFIG="attention-ablation-real-100m"
ATTENTION_TARGET_STEPS="$ATTENTION_EQUAL_TOKEN_STEPS"
ATTENTION_TARGET_SEQUENCE_LENGTH="$ATTENTION_SEQUENCE_LENGTH"
ATTENTION_TARGET_SEQUENCE_BATCH="$ATTENTION_EFFECTIVE_SEQUENCE_BATCH"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND MODEL_CONFIG
export ATTENTION_TARGET_STEPS ATTENTION_TARGET_SEQUENCE_LENGTH
export ATTENTION_TARGET_SEQUENCE_BATCH
setup_attention_runtime

DATASET_PATH="$(realpath "${DATASET_PATH:?Set DATASET_PATH to the prepared FineWeb-Edu DatasetDict directory.}")"
if [[ ! -f "$DATASET_PATH/dataset_dict.json" \
    || ! -f "$DATASET_PATH/optibertneo_manifest.json" ]]; then
    echo "FineWeb-Edu data is incomplete at $DATASET_PATH" >&2
    echo "Run jobs/multispace_fineweb/prepare_data.sbatch first." >&2
    exit 2
fi

RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$NEOBERT_ROOT/logs/real_fineweb}")"
EXPERIMENT_ID="${EXPERIMENT_ID:-fineweb-edu-s1024-real-100m-v1}"
SEED="${SEED:-42}"
SEGMENT_INDEX="${SEGMENT_INDEX:-1}"
MAX_STEPS="${MAX_STEPS:-84000}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5469}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-14000}"
EVAL_STEPS="${EVAL_STEPS:-14000}"
LOG_INTERVAL="${LOG_INTERVAL:-840}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-53640}"

validate_experiment_id "$EXPERIMENT_ID"
validate_attention_seed "$SEED"
for integer_name in \
    SEGMENT_INDEX MAX_STEPS MICRO_BATCH GRAD_ACCUM WARMUP_STEPS \
    CHECKPOINT_STEPS EVAL_STEPS LOG_INTERVAL MAX_TIME_SECONDS; do
    integer_value="${!integer_name}"
    if [[ ! "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$integer_name must be a positive integer; got '$integer_value'." >&2
        exit 2
    fi
done
if (( MICRO_BATCH * GRAD_ACCUM != ATTENTION_TARGET_SEQUENCE_BATCH )); then
    echo "MICRO_BATCH * GRAD_ACCUM must remain $ATTENTION_TARGET_SEQUENCE_BATCH; got $MICRO_BATCH * $GRAD_ACCUM." >&2
    exit 2
fi

RUNS_ROOT="$RUNS_ROOT/$EXPERIMENT_ID"
RUN_ROOT="$RUNS_ROOT/$ATTENTION_VARIANT/seed-$SEED"
WANDB_RUN_ID="ca-${EXPERIMENT_ID}-${ATTENTION_VARIANT}-s${SEED}"
mkdir -p "$RUN_ROOT" "${WANDB_DIR:-$RUNS_ROOT/wandb}"
export WANDB_DIR="${WANDB_DIR:-$RUNS_ROOT/wandb}"

echo "FineWeb-Edu real-attention segment $SEGMENT_INDEX"
echo "dataset=$DATASET_PATH"
echo "run_root=$RUN_ROOT"
echo "steps=$MAX_STEPS micro_batch=$MICRO_BATCH gradient_accumulation=$GRAD_ACCUM"
echo "max_time_seconds=$MAX_TIME_SECONDS (resume=${RUN_ROOT}/checkpoints)"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/validate_100m_pair.py
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space real \
    --backend flash \
    --require-a100

"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/train.py \
    dataset=fineweb_edu_google_1024 \
    tokenizer=google-1024 \
    model=attention-ablation-real-100m \
    model.attention_backend=flash \
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
    wandb.group="complex-attention-fineweb-edu-$EXPERIMENT_ID-seed-$SEED" \
    wandb.tags="[fineweb-edu,sample-10BT,sequence-1024,padding-free-packing,real-flash,100m]" \
    wandb.entity="${WANDB_ENTITY:-hyper_attention}" \
    wandb.project="${WANDB_PROJECT:-complex-attention-fineweb}" \
    wandb.mode="${WANDB_MODE:-online}"
