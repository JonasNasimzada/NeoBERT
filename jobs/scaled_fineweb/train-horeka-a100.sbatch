#!/usr/bin/env bash
#SBATCH --job-name=ca-scaled-fw-a100
#SBATCH --account=hk-project-pai00130
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# One resumable FineWeb-Edu segment for one of the four scaled models. The
# submit-horeka-training.sh launcher supplies MODEL_SPACE, MODEL_SIZE, and the
# dependency chain; this file is also directly runnable with those variables.
export SOURCE="${SOURCE:-/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention}"
export NEOBERT_ROOT="${NEOBERT_ROOT:-$SOURCE/NeoBERT}"
export COMPLEX_ATTENTION_ROOT="${COMPLEX_ATTENTION_ROOT:-$SOURCE}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hkfs/home/project/hk-project-pai00051/st_st171793}"
export MAMBA_ENV="${MAMBA_ENV:-$MAMBA_ROOT_PREFIX/envs/attention_dev}"
export MAMBA_EXE="${MAMBA_EXE:-$MAMBA_ROOT_PREFIX/micromamba/micromamba}"

MODEL_SPACE="${MODEL_SPACE:?Set MODEL_SPACE to real or multispace}"
MODEL_SIZE="${MODEL_SIZE:?Set MODEL_SIZE to 200m or 300m}"
case "$MODEL_SPACE" in
    real|multispace) ;;
    *) echo "MODEL_SPACE must be real or multispace; got '$MODEL_SPACE'." >&2; exit 2 ;;
esac
case "$MODEL_SIZE" in
    200m) expected_parameters=202043136 ;;
    300m) expected_parameters=304101120 ;;
    *) echo "MODEL_SIZE must be 200m or 300m; got '$MODEL_SIZE'." >&2; exit 2 ;;
esac

for required_path in \
    "$NEOBERT_ROOT/scripts/attention_ablation/train.py" \
    "$NEOBERT_ROOT/scripts/attention_ablation/validate_scaled_pairs.py" \
    "$MAMBA_EXE" \
    "$MAMBA_ENV/bin/python"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required HoreKa path is missing: $required_path" >&2
        exit 2
    fi
done

module purge
module load compiler/gnu/13
module load devel/cuda/12.9
module list
nvcc --version
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate "$MAMBA_ENV"
export COMPLEX_ATTN_PYTHON="$MAMBA_ENV/bin/python"
if [[ "$(realpath -e "$(command -v python)")" != "$(realpath -e "$COMPLEX_ATTN_PYTHON")" ]]; then
    echo "micromamba activated $(command -v python), expected $COMPLEX_ATTN_PYTHON" >&2
    exit 2
fi

export PYTHONPATH="$NEOBERT_ROOT/src:$COMPLEX_ATTENTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SLURM_TMPDIR="${TMPDIR:?HoreKa did not provide its node-local TMPDIR}"
cache_root="$TMPDIR/complex-attention-${SLURM_JOB_ID:-manual}"
mkdir -p "$cache_root/triton" "$cache_root/inductor" "$cache_root/cuda"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
export CUDA_CACHE_PATH="$cache_root/cuda"
export HF_HOME="${HF_HOME:-$COMPLEX_ATTENTION_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

DATASET_PATH="$(realpath "${DATASET_PATH:?Set DATASET_PATH to the prepared FineWeb-Edu DatasetDict directory}")"
if [[ ! -f "$DATASET_PATH/dataset_dict.json" \
    || ! -f "$DATASET_PATH/optibertneo_manifest.json" ]]; then
    echo "FineWeb-Edu data is incomplete at $DATASET_PATH" >&2
    exit 2
fi

RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$NEOBERT_ROOT/logs/scaled_fineweb}")"
EXPERIMENT_ID="${EXPERIMENT_ID:-fineweb-edu-s1024-$MODEL_SIZE-v1}"
SEED="${SEED:-42}"
SEGMENT_INDEX="${SEGMENT_INDEX:-1}"
MAX_STEPS="${MAX_STEPS:-84000}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-5469}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-14000}"
EVAL_STEPS="${EVAL_STEPS:-14000}"
LOG_INTERVAL="${LOG_INTERVAL:-840}"
# Leave 30 minutes inside HoreKa's two-day walltime for a checkpoint/export.
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-169200}"

if (( MICRO_BATCH * GRAD_ACCUM != 16 )); then
    echo "MICRO_BATCH * GRAD_ACCUM must remain 16; got $MICRO_BATCH * $GRAD_ACCUM." >&2
    exit 2
fi
for integer_name in \
    SEGMENT_INDEX MAX_STEPS MICRO_BATCH GRAD_ACCUM WARMUP_STEPS \
    CHECKPOINT_STEPS EVAL_STEPS LOG_INTERVAL MAX_TIME_SECONDS; do
    integer_value="${!integer_name}"
    if [[ ! "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$integer_name must be a positive integer; got '$integer_value'." >&2
        exit 2
    fi
done

RUNS_ROOT="$RUNS_ROOT/$EXPERIMENT_ID"
ATTENTION_VARIANT="$MODEL_SPACE-flash"
MODEL_CONFIG="attention-ablation-$MODEL_SPACE-$MODEL_SIZE"
RUN_ROOT="$RUNS_ROOT/$ATTENTION_VARIANT/seed-$SEED"
WANDB_RUN_ID="ca-${EXPERIMENT_ID}-${ATTENTION_VARIANT}-s${SEED}"
mkdir -p "$RUN_ROOT" "${WANDB_DIR:-$RUNS_ROOT/wandb}"
export WANDB_DIR="${WANDB_DIR:-$RUNS_ROOT/wandb}"

echo "FineWeb-Edu HoreKa segment $SEGMENT_INDEX"
echo "model=$MODEL_CONFIG parameters=$expected_parameters"
echo "dataset=$DATASET_PATH"
echo "run_root=$RUN_ROOT"
echo "steps=$MAX_STEPS micro_batch=$MICRO_BATCH gradient_accumulation=$GRAD_ACCUM"
echo "max_time_seconds=$MAX_TIME_SECONDS"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/validate_scaled_pairs.py \
    --size "$MODEL_SIZE" --require-a100
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "$MODEL_SPACE" --backend flash --require-a100

"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/train.py \
    dataset=fineweb_edu_google_1024 \
    tokenizer=google-1024 \
    model="$MODEL_CONFIG" \
    model.attention_backend=flash \
    dataset.path_to_disk="$DATASET_PATH" \
    dataset.pack_to_length=1024 \
    tokenizer.max_length=1024 \
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
    dataloader.train.max_length=1024 \
    dataloader.validation.batch_size="$MICRO_BATCH" \
    dataloader.validation.max_length=1024 \
    seed="$SEED" \
    wandb.log_interval="$LOG_INTERVAL" \
    wandb.id="$WANDB_RUN_ID" \
    wandb.name="$EXPERIMENT_ID-$ATTENTION_VARIANT-seed-$SEED" \
    wandb.group="complex-attention-fineweb-edu-$EXPERIMENT_ID-seed-$SEED" \
    wandb.tags="[fineweb-edu,sample-1p6B,sequence-1024,padding-free-packing,$ATTENTION_VARIANT,$MODEL_SIZE,parameter-matched]" \
    wandb.entity="${WANDB_ENTITY:-hyper_attention}" \
    wandb.project="${WANDB_PROJECT:-complex-attention-fineweb}" \
    wandb.mode="${WANDB_MODE:-online}"
