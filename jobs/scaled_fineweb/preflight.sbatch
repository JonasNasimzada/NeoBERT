#!/usr/bin/env bash
#SBATCH --job-name=ca-scaled-model-test
#SBATCH --partition=fastlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=ca-scaled-model-test-%A_%a.out
#SBATCH --error=ca-scaled-model-test-%A_%a.err

set -euo pipefail

# TEST ONLY: this allocation never loads FineWeb-Edu, enters the pretraining
# loop, writes a checkpoint, or submits another Slurm job.
if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/scaled_fineweb"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/scaled_fineweb"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
common_file="$job_dir/../attention_ablation/common.sh"
if [[ ! -f "$common_file" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    exit 2
fi

source "$common_file"
export ATTENTION_SPACE=multispace
export ATTENTION_BACKEND=flash
case "${ATTENTION_RUNTIME_PRECONFIGURED:-0}" in
    0)
        setup_attention_runtime
        ;;
    1)
        : "${NEOBERT_ROOT:?A preconfigured runtime must export NEOBERT_ROOT}"
        : "${COMPLEX_ATTENTION_ROOT:?A preconfigured runtime must export COMPLEX_ATTENTION_ROOT}"
        : "${COMPLEX_ATTN_PYTHON:?A preconfigured runtime must export COMPLEX_ATTN_PYTHON}"
        if [[ ! -x "$COMPLEX_ATTN_PYTHON" ]]; then
            echo "Preconfigured Python is not executable: $COMPLEX_ATTN_PYTHON" >&2
            exit 2
        fi
        ;;
    *)
        echo "ATTENTION_RUNTIME_PRECONFIGURED must be 0 or 1." >&2
        exit 2
        ;;
esac
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

task_id="${SLURM_ARRAY_TASK_ID:-0}"
sizes=(200m 200m 300m 300m)
variants=(real multispace real multispace)
layers=(21 21 33 33)
if [[ ! "$task_id" =~ ^[0-3]$ ]]; then
    echo "SLURM_ARRAY_TASK_ID must be 0 through 3; got '$task_id'." >&2
    exit 2
fi
size="${sizes[$task_id]}"
variant="${variants[$task_id]}"
layer_count="${layers[$task_id]}"

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
PREFLIGHT_ROOT="$(realpath -m "${PREFLIGHT_ROOT:-$NEOBERT_ROOT/logs/scaled_fineweb/preflight-$array_job_id}")"
mkdir -p "$PREFLIGHT_ROOT"

cd "$NEOBERT_ROOT"

# Static checks are deliberately inside the A100 allocation too, keeping every
# validation command under the project's GPU-only test policy. Task zero owns
# the shared checks; the four production graphs run concurrently below.
if [[ "$task_id" == 0 ]]; then
    git diff --check -- \
        conf/model/attention-ablation-real-200m.yaml \
        conf/model/attention-ablation-multispace-200m.yaml \
        conf/model/attention-ablation-real-300m.yaml \
        conf/model/attention-ablation-multispace-300m.yaml \
        scripts/attention_ablation/validate_scaled_pairs.py \
        scripts/attention_ablation/smoke_scaled_model.py \
        tests/test_scaled_attention_models.py \
        jobs/scaled_fineweb/preflight.sbatch \
        jobs/scaled_fineweb/preflight-horeka-a100.sbatch \
        jobs/scaled_fineweb/train.sbatch \
        jobs/scaled_fineweb/submit.sh \
        SCALED_ATTENTION_MODELS.md
    bash -n \
        jobs/scaled_fineweb/preflight.sbatch \
        jobs/scaled_fineweb/train.sbatch \
        jobs/scaled_fineweb/submit.sh
    DRY_RUN=1 TRAIN_SEGMENTS=1 \
        bash jobs/scaled_fineweb/submit.sh all \
        >"$PREFLIGHT_ROOT/submit-dry-run.txt"
    "$COMPLEX_ATTN_PYTHON" -m py_compile \
        scripts/attention_ablation/validate_scaled_pairs.py \
        scripts/attention_ablation/smoke_scaled_model.py
    "$COMPLEX_ATTN_PYTHON" -m pytest -q -p no:cacheprovider \
        tests/test_scaled_attention_models.py
fi

"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/validate_scaled_pairs.py \
    --size "$size" \
    --require-a100 \
    --output "$PREFLIGHT_ROOT/scaled-pairs-$size-$variant.json"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "$variant" --backend flash --require-a100

run_smoke() {
    local size="${1:?model size is required}"
    local variant="${2:?variant is required}"
    local layers="${3:?layer count is required}"
    local sequence_length="${4:?sequence length is required}"
    local batch_size="${5:-1}"
    local label="$size-$variant-l${layers}-s${sequence_length}-b${batch_size}"

    echo "GPU smoke: size=$size variant=$variant layers=$layers sequence_length=$sequence_length batch_size=$batch_size"
    "$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/smoke_scaled_model.py \
        --size "$size" \
        --variant "$variant" \
        --layers "$layers" \
        --sequence-length "$sequence_length" \
        --batch-size "$batch_size" \
        --require-a100 \
        --output "$PREFLIGHT_ROOT/$label.json"
}

# Each array task owns one requested model. Together the four A100 tasks run in
# parallel: full production depth at a conservative context, then the exact
# 1,024-token FlashAttention kernel geometry at one production-width layer.
run_smoke "$size" "$variant" "$layer_count" 64
run_smoke "$size" "$variant" 1 1024

case "${FULL_PRODUCTION_GEOMETRY:-0}" in
    0) ;;
    1)
        # Optional decisive checks; these are still single synthetic optimizer
        # steps and never enter the FineWeb-Edu training loop.
        production_batch=1
        if [[ "$variant" == multispace ]]; then
            production_batch="${MULTISPACE_PRODUCTION_BATCH:-1}"
        fi
        run_smoke "$size" "$variant" "$layer_count" 1024 "$production_batch"
        ;;
    *)
        echo "FULL_PRODUCTION_GEOMETRY must be 0 or 1." >&2
        exit 2
        ;;
esac

echo "Scaled $size $variant A100 validation passed: $PREFLIGHT_ROOT"
echo "No FineWeb-Edu data was loaded and no training job was submitted."
