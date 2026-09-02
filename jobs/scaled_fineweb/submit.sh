#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: jobs/scaled_fineweb/submit.sh [selection]

Selections:
  all                 real and multispace at 200M and 300M (default)
  200m | 300m         both attention variants at one size
  real | multispace   both sizes for one attention variant
  real-200m | multispace-200m | real-300m | multispace-300m
  preflight           submit only the A100 validation job

The launcher defaults to DRY_RUN=1. A live training submission requires both
DRY_RUN=0 and CONFIRM_FULL_SUBMISSION=YES. A live preflight-only submission is
safe and needs only DRY_RUN=0.

Optional environment:
  DATASET_PATH         Prepared FineWeb-Edu DatasetDict directory.
  RUNS_ROOT            Parent directory for all four stable run roots.
  EXPERIMENT_PREFIX    Default: fineweb-edu-s1024 (size and v1 are appended).
  SEED                  Default: 42.
  TRAIN_SEGMENTS       Sequential 15-hour jobs per model (default: 5).
  SKIP_PREP            Set to 1 after validating an existing dataset.
  PREP_JOB_ID          Existing preparation job dependency.
  SKIP_PREFLIGHT       Set to 1 to omit the validation dependency.
  PREFLIGHT_JOB_ID     Existing A100 preflight dependency.
  FULL_PRODUCTION_GEOMETRY
                        Set to 1 to add full-depth 1,024-token smoke steps.
  MAX_STEPS            Default: 84000 equal-token optimizer steps.
  MICRO_BATCH          Default: 2.
  GRAD_ACCUM           Default: 8 (effective batch remains 16).
  SBATCH_EXTRA_ARGS    Whitespace-separated site-specific sbatch flags.
EOF
    exit 2
}

if (($# > 1)); then
    usage
fi

selection="${1:-all}"
case "$selection" in
    all)
        models=(real-200m multispace-200m real-300m multispace-300m)
        ;;
    200m)
        models=(real-200m multispace-200m)
        ;;
    300m)
        models=(real-300m multispace-300m)
        ;;
    real)
        models=(real-200m real-300m)
        ;;
    multispace)
        models=(multispace-200m multispace-300m)
        ;;
    real-200m|multispace-200m|real-300m|multispace-300m)
        models=("$selection")
        ;;
    preflight)
        models=()
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown scaled-model selection: $selection" >&2
        usage
        ;;
esac

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
neobert_root="$(cd "$job_dir/../.." && pwd)"
complex_attention_root="$(cd "$neobert_root/.." && pwd)"
common_file="$job_dir/../attention_ablation/common.sh"
prepare_job="$job_dir/../multispace_fineweb/prepare_data.sbatch"
preflight_job="$job_dir/preflight.sbatch"
train_job="$job_dir/train.sbatch"
if [[ ! -f "$common_file" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    exit 2
fi
source "$common_file"

DATASET_PATH="$(realpath -m "${DATASET_PATH:-$neobert_root/tokenized_datasets/fineweb_edu_google_1024_1p6b}")"
RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$neobert_root/logs/scaled_fineweb}")"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-fineweb-edu-s1024}"
SEED="${SEED:-42}"
TRAIN_SEGMENTS="${TRAIN_SEGMENTS:-5}"
DRY_RUN="${DRY_RUN:-1}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
PREP_JOB_ID="${PREP_JOB_ID:-}"
PREFLIGHT_JOB_ID="${PREFLIGHT_JOB_ID:-}"
FULL_PRODUCTION_GEOMETRY="${FULL_PRODUCTION_GEOMETRY:-0}"
MAX_STEPS="${MAX_STEPS:-84000}"
MICRO_BATCH="${MICRO_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-5469}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-14000}"
EVAL_STEPS="${EVAL_STEPS:-14000}"
LOG_INTERVAL="${LOG_INTERVAL:-840}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-53640}"

validate_experiment_id "$EXPERIMENT_PREFIX"
validate_attention_seed "$SEED"
for integer_name in \
    TRAIN_SEGMENTS MAX_STEPS MICRO_BATCH GRAD_ACCUM WARMUP_STEPS \
    CHECKPOINT_STEPS EVAL_STEPS LOG_INTERVAL MAX_TIME_SECONDS; do
    integer_value="${!integer_name}"
    if [[ ! "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$integer_name must be a positive integer; got '$integer_value'." >&2
        exit 2
    fi
done
if (( MICRO_BATCH * GRAD_ACCUM != ATTENTION_EFFECTIVE_SEQUENCE_BATCH )); then
    echo "MICRO_BATCH * GRAD_ACCUM must remain $ATTENTION_EFFECTIVE_SEQUENCE_BATCH; got $MICRO_BATCH * $GRAD_ACCUM." >&2
    exit 2
fi
for boolean_name in DRY_RUN SKIP_PREP SKIP_PREFLIGHT FULL_PRODUCTION_GEOMETRY; do
    boolean_value="${!boolean_name}"
    if [[ "$boolean_value" != 0 && "$boolean_value" != 1 ]]; then
        echo "$boolean_name must be 0 or 1; got '$boolean_value'." >&2
        exit 2
    fi
done
for job_id_name in PREP_JOB_ID PREFLIGHT_JOB_ID; do
    job_id_value="${!job_id_name}"
    if [[ -n "$job_id_value" && ! "$job_id_value" =~ ^[0-9]+$ ]]; then
        echo "$job_id_name must be a numeric Slurm job id; got '$job_id_value'." >&2
        exit 2
    fi
done
if [[ -n "$PREP_JOB_ID" && "$SKIP_PREP" == 1 ]]; then
    echo "Set either PREP_JOB_ID or SKIP_PREP=1, not both." >&2
    exit 2
fi
if [[ -n "$PREFLIGHT_JOB_ID" && "$SKIP_PREFLIGHT" == 1 ]]; then
    echo "Set either PREFLIGHT_JOB_ID or SKIP_PREFLIGHT=1, not both." >&2
    exit 2
fi
if [[ "$DRY_RUN" == 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; use a Slurm login node or set DRY_RUN=1." >&2
    exit 2
fi
if [[ "$DRY_RUN" == 0 && "$selection" != preflight \
    && "${CONFIRM_FULL_SUBMISSION:-}" != YES ]]; then
    echo "Live training requires DRY_RUN=0 and CONFIRM_FULL_SUBMISSION=YES." >&2
    exit 2
fi
for exported_value in \
    "$DATASET_PATH" "$RUNS_ROOT" "$neobert_root" \
    "$complex_attention_root" "$EXPERIMENT_PREFIX"; do
    if [[ "$exported_value" == *","* || "$exported_value" == *$'\n'* ]]; then
        echo "Slurm-exported values may not contain commas or newlines: $exported_value" >&2
        exit 2
    fi
done
for required_job in "$preflight_job" "$train_job"; do
    if [[ ! -f "$required_job" ]]; then
        echo "Required job file is missing: $required_job" >&2
        exit 2
    fi
done
if [[ "$selection" != preflight && "$SKIP_PREP" == 0 && -z "$PREP_JOB_ID" \
    && ! -f "$prepare_job" ]]; then
    echo "FineWeb-Edu preparation job is missing: $prepare_job" >&2
    exit 2
fi
if [[ "$selection" != preflight && "$SKIP_PREP" == 1 ]]; then
    if [[ ! -f "$DATASET_PATH/dataset_dict.json" \
        || ! -f "$DATASET_PATH/optibertneo_manifest.json" ]]; then
        echo "SKIP_PREP=1 requires a complete prepared DatasetDict at $DATASET_PATH" >&2
        exit 2
    fi
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

preflight_dependency="$PREFLIGHT_JOB_ID"
if [[ -z "$preflight_dependency" && "$SKIP_PREFLIGHT" == 0 ]]; then
    preflight_command=(
        "${sbatch_extra_args[@]}"
        --chdir="$neobert_root"
        --export="ALL,NEOBERT_ROOT=$neobert_root,COMPLEX_ATTENTION_ROOT=$complex_attention_root,FULL_PRODUCTION_GEOMETRY=$FULL_PRODUCTION_GEOMETRY"
        "$preflight_job"
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "A100 scaled-model preflight:"
        print_command sbatch --parsable "${preflight_command[@]}"
        preflight_dependency=PREFLIGHT_JOB_ID
    else
        preflight_dependency="$(submit_job "${preflight_command[@]}")"
        echo "A100 scaled-model preflight: $preflight_dependency"
    fi
elif [[ -n "$preflight_dependency" ]]; then
    echo "A100 scaled-model preflight dependency: $preflight_dependency"
else
    echo "A100 scaled-model preflight: explicitly skipped"
fi

if [[ "$selection" == preflight ]]; then
    echo "Preflight only; no preparation or training job was submitted."
    exit 0
fi

prep_dependency="$PREP_JOB_ID"
if [[ -z "$prep_dependency" && "$SKIP_PREP" == 0 ]]; then
    prep_command=(
        "${sbatch_extra_args[@]}"
        --chdir="$neobert_root"
        --export="ALL,NEOBERT_ROOT=$neobert_root,COMPLEX_ATTENTION_ROOT=$complex_attention_root,DATASET_PATH=$DATASET_PATH"
        "$prepare_job"
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "FineWeb-Edu preparation:"
        print_command sbatch --parsable "${prep_command[@]}"
        prep_dependency=PREP_JOB_ID
    else
        prep_dependency="$(submit_job "${prep_command[@]}")"
        echo "FineWeb-Edu preparation: $prep_dependency"
    fi
elif [[ -n "$prep_dependency" ]]; then
    echo "FineWeb-Edu preparation dependency: $prep_dependency"
else
    echo "FineWeb-Edu preparation: skipped (validated $DATASET_PATH)"
fi

initial_dependency_ids=()
if [[ -n "$preflight_dependency" ]]; then
    initial_dependency_ids+=("$preflight_dependency")
fi
if [[ -n "$prep_dependency" ]]; then
    initial_dependency_ids+=("$prep_dependency")
fi
initial_dependency=""
if ((${#initial_dependency_ids[@]} > 0)); then
    initial_dependency="$(IFS=:; printf '%s' "${initial_dependency_ids[*]}")"
fi

all_training_ids=()
for model in "${models[@]}"; do
    model_space="${model%-*}"
    model_size="${model##*-}"
    experiment_id="$EXPERIMENT_PREFIX-$model_size-v1"
    previous_job_id=""
    model_training_ids=()

    for ((segment_index = 1; segment_index <= TRAIN_SEGMENTS; segment_index++)); do
        dependency_args=()
        if [[ -n "$previous_job_id" ]]; then
            dependency_args=(--dependency="afterok:$previous_job_id")
        elif [[ -n "$initial_dependency" ]]; then
            dependency_args=(--dependency="afterok:$initial_dependency")
        fi
        train_command=(
            "${sbatch_extra_args[@]}"
            --job-name="ca-fw-$model"
            --chdir="$neobert_root"
            "${dependency_args[@]}"
            --export="ALL,NEOBERT_ROOT=$neobert_root,COMPLEX_ATTENTION_ROOT=$complex_attention_root,DATASET_PATH=$DATASET_PATH,RUNS_ROOT=$RUNS_ROOT,EXPERIMENT_ID=$experiment_id,MODEL_SPACE=$model_space,MODEL_SIZE=$model_size,SEED=$SEED,SEGMENT_INDEX=$segment_index,MAX_STEPS=$MAX_STEPS,MICRO_BATCH=$MICRO_BATCH,GRAD_ACCUM=$GRAD_ACCUM,WARMUP_STEPS=$WARMUP_STEPS,CHECKPOINT_STEPS=$CHECKPOINT_STEPS,EVAL_STEPS=$EVAL_STEPS,LOG_INTERVAL=$LOG_INTERVAL,MAX_TIME_SECONDS=$MAX_TIME_SECONDS"
            "$train_job"
        )
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "$model training segment $segment_index:"
            print_command sbatch --parsable "${train_command[@]}"
            previous_job_id="${model^^}_SEGMENT_${segment_index}_JOB_ID"
            previous_job_id="${previous_job_id//-/_}"
        else
            previous_job_id="$(submit_job "${train_command[@]}")"
            echo "$model training segment $segment_index: $previous_job_id"
        fi
        model_training_ids+=("$previous_job_id")
        all_training_ids+=("$previous_job_id")
    done

    echo "$model run root: $RUNS_ROOT/$experiment_id/$model_space-flash/seed-$SEED"
    echo "$model chain: ${model_training_ids[*]}"
done

echo "Dataset: $DATASET_PATH"
echo "Equal-token schedule: $MAX_STEPS steps x $ATTENTION_EFFECTIVE_SEQUENCE_BATCH sequences x $ATTENTION_SEQUENCE_LENGTH tokens"
echo "All training jobs: ${all_training_ids[*]}"
