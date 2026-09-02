#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: jobs/scaled_fineweb/submit-horeka-training.sh [selection]

Selections: all (default), 200m, 300m, real, multispace,
real-200m, multispace-200m, real-300m, multispace-300m, preflight.

The default is DRY_RUN=1. A live training submission requires
DRY_RUN=0 CONFIRM_FULL_SUBMISSION=YES. The default upstream dependency is
5125968; set UPSTREAM_JOB_ID= to omit it only after checking that job's result.

Required for live training:
  DATASET_PATH   Prepared FineWeb-Edu DatasetDict directory.

Useful overrides:
  SOURCE, MAMBA_ENV, MAMBA_EXE, MAMBA_ROOT_PREFIX
  PREFLIGHT_JOB_ID, SKIP_PREFLIGHT, TRAIN_SEGMENTS, MAX_STEPS
  MICRO_BATCH (default 1), GRAD_ACCUM (default 16), RUNS_ROOT, SEED
EOF
    exit 2
}

if (($# > 1)); then
    usage
fi

selection="${1:-all}"
case "$selection" in
    all) models=(real-200m multispace-200m real-300m multispace-300m) ;;
    200m) models=(real-200m multispace-200m) ;;
    300m) models=(real-300m multispace-300m) ;;
    real) models=(real-200m real-300m) ;;
    multispace) models=(multispace-200m multispace-300m) ;;
    real-200m|multispace-200m|real-300m|multispace-300m) models=("$selection") ;;
    preflight) models=() ;;
    -h|--help) usage ;;
    *) echo "Unknown selection: $selection" >&2; usage ;;
esac

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source="${SOURCE:-/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention}"
neobert_root="${NEOBERT_ROOT:-$source/NeoBERT}"
complex_attention_root="${COMPLEX_ATTENTION_ROOT:-$source}"
account="${HOREKA_ACCOUNT:-hk-project-pai00130}"
dataset_path="$(realpath -m "${DATASET_PATH:-$neobert_root/tokenized_datasets/fineweb_edu_google_1024_1p6b}")"
runs_root="$(realpath -m "${RUNS_ROOT:-$neobert_root/logs/scaled_fineweb}")"
experiment_prefix="${EXPERIMENT_PREFIX:-fineweb-edu-s1024}"
# Use `-` (not `:-`) so UPSTREAM_JOB_ID= intentionally disables the
# dependency after the user has verified that the predecessor completed.
upstream_job_id="${UPSTREAM_JOB_ID-5125968}"
preflight_job_id="${PREFLIGHT_JOB_ID:-}"
skip_preflight="${SKIP_PREFLIGHT:-0}"
dry_run="${DRY_RUN:-1}"
confirm="${CONFIRM_FULL_SUBMISSION:-}"
# The training launcher gates the selected A100-40 microbatch at full context
# by default. Set FULL_PRODUCTION_GEOMETRY=0 for the shorter structural gate.
full_production_geometry="${FULL_PRODUCTION_GEOMETRY:-1}"
train_segments="${TRAIN_SEGMENTS:-5}"
max_steps="${MAX_STEPS:-84000}"
micro_batch="${MICRO_BATCH:-1}"
grad_accum="${GRAD_ACCUM:-16}"
warmup_steps="${WARMUP_STEPS:-5469}"
checkpoint_steps="${CHECKPOINT_STEPS:-14000}"
eval_steps="${EVAL_STEPS:-14000}"
log_interval="${LOG_INTERVAL:-840}"
max_time_seconds="${MAX_TIME_SECONDS:-169200}"
seed="${SEED:-42}"
mamba_root_prefix="${MAMBA_ROOT_PREFIX:-/hkfs/home/project/hk-project-pai00051/st_st171793}"
mamba_env="${MAMBA_ENV:-$mamba_root_prefix/envs/attention_dev}"
mamba_exe="${MAMBA_EXE:-$mamba_root_prefix/micromamba/micromamba}"

if [[ ! "$account" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "HOREKA_ACCOUNT contains unsupported characters: $account" >&2
    exit 2
fi
if [[ ! "$experiment_prefix" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EXPERIMENT_PREFIX may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi
for integer_name in train_segments max_steps micro_batch grad_accum warmup_steps \
    checkpoint_steps eval_steps log_interval max_time_seconds seed; do
    integer_value="${!integer_name}"
    if [[ ! "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$integer_name must be a positive integer; got '$integer_value'." >&2
        exit 2
    fi
done
if (( micro_batch * grad_accum != 16 )); then
    echo "MICRO_BATCH * GRAD_ACCUM must remain 16; got $micro_batch * $grad_accum." >&2
    exit 2
fi
for boolean_name in dry_run skip_preflight full_production_geometry; do
    boolean_value="${!boolean_name}"
    if [[ "$boolean_value" != 0 && "$boolean_value" != 1 ]]; then
        echo "$boolean_name must be 0 or 1; got '$boolean_value'." >&2
        exit 2
    fi
done
for job_id_name in upstream_job_id preflight_job_id; do
    job_id_value="${!job_id_name}"
    if [[ -n "$job_id_value" && ! "$job_id_value" =~ ^[0-9]+$ ]]; then
        echo "$job_id_name must be numeric or empty; got '$job_id_value'." >&2
        exit 2
    fi
done
if [[ "$dry_run" == 0 && "$selection" != preflight \
    && "$confirm" != YES ]]; then
    echo "Live training requires DRY_RUN=0 and CONFIRM_FULL_SUBMISSION=YES." >&2
    exit 2
fi
if [[ "$skip_preflight" == 1 && -n "$preflight_job_id" ]]; then
    echo "Set either SKIP_PREFLIGHT=1 or PREFLIGHT_JOB_ID, not both." >&2
    exit 2
fi
if [[ "$selection" != preflight && "$dry_run" == 0 ]]; then
    if [[ ! -f "$dataset_path/dataset_dict.json" \
        || ! -f "$dataset_path/optibertneo_manifest.json" ]]; then
        echo "Prepared FineWeb-Edu DatasetDict is incomplete: $dataset_path" >&2
        exit 2
    fi
fi
for exported_value in "$source" "$neobert_root" "$complex_attention_root" \
    "$dataset_path" "$runs_root" "$experiment_prefix"; do
    if [[ "$exported_value" == *,* || "$exported_value" == *$'\n'* ]]; then
        echo "Slurm-exported values may not contain commas or newlines: $exported_value" >&2
        exit 2
    fi
done
for required_job in "$job_dir/preflight-horeka-a100.sbatch" \
    "$job_dir/train-horeka-a100.sbatch"; do
    if [[ ! -f "$required_job" ]]; then
        echo "Required job file is missing: $required_job" >&2
        exit 2
    fi
done
if [[ "$dry_run" == 0 && ! -x "$mamba_exe" ]]; then
    echo "MAMBA_EXE is not executable: $mamba_exe" >&2
    exit 2
fi
if [[ "$dry_run" == 0 && "$selection" != preflight && ! -x "$mamba_env/bin/python" ]]; then
    echo "Micromamba Python is not executable: $mamba_env/bin/python" >&2
    exit 2
fi
if [[ "$dry_run" == 0 && ! -d "$neobert_root" ]]; then
    echo "NEOBERT_ROOT does not exist: $neobert_root" >&2
    exit 2
fi
if [[ "$dry_run" == 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; run this command on a HoreKa login node." >&2
    exit 2
fi

read -r -a sbatch_extra_args <<<"${SBATCH_EXTRA_ARGS:-}"
print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}
submit_job() {
    local result
    result="$(sbatch --parsable "$@")"
    result="${result%%;*}"
    if [[ ! "$result" =~ ^[0-9]+$ ]]; then
        echo "sbatch returned an invalid job id: $result" >&2
        return 2
    fi
    printf '%s\n' "$result"
}

preflight_dependency="$preflight_job_id"
if [[ "$skip_preflight" == 0 ]]; then
    if [[ -z "$preflight_dependency" ]]; then
        preflight_dependency_args=()
        if [[ -n "$upstream_job_id" ]]; then
            preflight_dependency_args=(
                --dependency="afterok:$upstream_job_id"
                --kill-on-invalid-dep=yes
            )
        fi
        preflight_command=(
            "${sbatch_extra_args[@]}"
            --account="$account"
            --partition=accelerated
            --chdir="$neobert_root"
            "${preflight_dependency_args[@]}"
            --export="ALL,SOURCE=$source,NEOBERT_ROOT=$neobert_root,COMPLEX_ATTENTION_ROOT=$complex_attention_root,MAMBA_ROOT_PREFIX=$mamba_root_prefix,MAMBA_ENV=$mamba_env,MAMBA_EXE=$mamba_exe,FULL_PRODUCTION_GEOMETRY=$full_production_geometry"
            "$job_dir/preflight-horeka-a100.sbatch"
        )
        if [[ "$dry_run" == 1 ]]; then
            echo "HoreKa A100 preflight array:"
            print_command sbatch --parsable "${preflight_command[@]}"
            preflight_dependency=PREFLIGHT_JOB_ID
        else
            preflight_dependency="$(submit_job "${preflight_command[@]}")"
            echo "HoreKa A100 preflight array: $preflight_dependency"
        fi
    else
        echo "Using existing HoreKa preflight dependency: $preflight_dependency"
    fi
fi

if [[ "$selection" == preflight ]]; then
    echo "Preflight only; no FineWeb-Edu training jobs submitted."
    exit 0
fi

initial_dependency=""
if [[ -n "$preflight_dependency" ]]; then
    initial_dependency="$preflight_dependency"
elif [[ -n "$upstream_job_id" ]]; then
    initial_dependency="$upstream_job_id"
fi

all_training_ids=()
for model in "${models[@]}"; do
    model_space="${model%-*}"
    model_size="${model##*-}"
    experiment_id="$experiment_prefix-$model_size-v1"
    previous_job_id=""
    model_training_ids=()
    for ((segment_index = 1; segment_index <= train_segments; segment_index++)); do
        dependency_args=()
        if [[ -n "$previous_job_id" ]]; then
            dependency_args=(--dependency="afterok:$previous_job_id")
        elif [[ -n "$initial_dependency" ]]; then
            dependency_args=(--dependency="afterok:$initial_dependency")
        fi
        train_command=(
            "${sbatch_extra_args[@]}"
            --account="$account"
            --partition=accelerated
            --nodes=1
            --ntasks=1
            --gres=gpu:1
            --cpus-per-task=12
            --time=2-00:00:00
            --job-name="ca-fw-$model"
            --chdir="$neobert_root"
            "${dependency_args[@]}"
            --export="ALL,SOURCE=$source,NEOBERT_ROOT=$neobert_root,COMPLEX_ATTENTION_ROOT=$complex_attention_root,MAMBA_ROOT_PREFIX=$mamba_root_prefix,MAMBA_ENV=$mamba_env,MAMBA_EXE=$mamba_exe,DATASET_PATH=$dataset_path,RUNS_ROOT=$runs_root,EXPERIMENT_ID=$experiment_id,MODEL_SPACE=$model_space,MODEL_SIZE=$model_size,SEED=$seed,SEGMENT_INDEX=$segment_index,MAX_STEPS=$max_steps,MICRO_BATCH=$micro_batch,GRAD_ACCUM=$grad_accum,WARMUP_STEPS=$warmup_steps,CHECKPOINT_STEPS=$checkpoint_steps,EVAL_STEPS=$eval_steps,LOG_INTERVAL=$log_interval,MAX_TIME_SECONDS=$max_time_seconds"
            "$job_dir/train-horeka-a100.sbatch"
        )
        if [[ "$dry_run" == 1 ]]; then
            echo "$model segment $segment_index:"
            print_command sbatch --parsable "${train_command[@]}"
            previous_job_id="${model^^}_SEGMENT_${segment_index}_JOB_ID"
            previous_job_id="${previous_job_id//-/_}"
        else
            previous_job_id="$(submit_job "${train_command[@]}")"
            echo "$model segment $segment_index: $previous_job_id"
        fi
        model_training_ids+=("$previous_job_id")
        all_training_ids+=("$previous_job_id")
    done
    echo "$model run root: $runs_root/$experiment_id/$model_space-flash/seed-$seed"
    echo "$model chain: ${model_training_ids[*]}"
done

echo "Dataset: $dataset_path"
echo "Schedule: $max_steps steps x 16 sequences/update x 1024 tokens"
echo "Training jobs: ${all_training_ids[*]}"
