#!/usr/bin/env bash

set -euo pipefail

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
neobert_root="$(cd "$job_dir/../.." && pwd)"
complex_attention_root="$(cd "$neobert_root/.." && pwd)"
common_file="$job_dir/../attention_ablation/common.sh"
if [[ ! -f "$common_file" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    exit 2
fi
source "$common_file"

DATASET_PATH="$(realpath -m "${DATASET_PATH:-$neobert_root/tokenized_datasets/fineweb_edu_google_1024_1p6b}")"
RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$neobert_root/logs/multispace_fineweb}")"
EXPERIMENT_ID="${EXPERIMENT_ID:-fineweb-edu-s1024-multispace-100m-v1}"
SEED="${SEED:-42}"
TRAIN_SEGMENTS="${TRAIN_SEGMENTS:-5}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREP="${SKIP_PREP:-0}"
PREP_JOB_ID="${PREP_JOB_ID:-}"

validate_experiment_id "$EXPERIMENT_ID"
validate_attention_seed "$SEED"
if [[ ! "$TRAIN_SEGMENTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_SEGMENTS must be a positive integer; got '$TRAIN_SEGMENTS'." >&2
    exit 2
fi
for boolean_name in DRY_RUN SKIP_PREP; do
    boolean_value="${!boolean_name}"
    if [[ "$boolean_value" != 0 && "$boolean_value" != 1 ]]; then
        echo "$boolean_name must be 0 or 1; got '$boolean_value'." >&2
        exit 2
    fi
done
if [[ "$DRY_RUN" == 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; run this launcher on a Slurm login node or set DRY_RUN=1." >&2
    exit 2
fi
if [[ -n "$PREP_JOB_ID" && ! "$PREP_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "PREP_JOB_ID must be a numeric Slurm job id; got '$PREP_JOB_ID'." >&2
    exit 2
fi
if [[ -n "$PREP_JOB_ID" && "$SKIP_PREP" == 1 ]]; then
    echo "Set either PREP_JOB_ID or SKIP_PREP=1, not both." >&2
    exit 2
fi
for exported_path in "$DATASET_PATH" "$RUNS_ROOT" "$neobert_root" "$complex_attention_root"; do
    if [[ "$exported_path" == *","* || "$exported_path" == *$'\n'* ]]; then
        echo "Slurm-exported paths may not contain commas or newlines: $exported_path" >&2
        exit 2
    fi
done
for required_job in "$job_dir/prepare_data.sbatch" "$job_dir/train.sbatch"; do
    if [[ ! -f "$required_job" ]]; then
        echo "Required job file is missing: $required_job" >&2
        exit 2
    fi
done

if [[ "$SKIP_PREP" == 1 ]]; then
    if [[ ! -f "$DATASET_PATH/dataset_dict.json" \
        || ! -f "$DATASET_PATH/optibertneo_manifest.json" ]]; then
        echo "SKIP_PREP=1 requires a complete prepared DatasetDict at $DATASET_PATH" >&2
        exit 2
    fi
fi

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

prep_dependency="$PREP_JOB_ID"
if [[ -z "$prep_dependency" && "$SKIP_PREP" == 0 ]]; then
    prep_command=(
        --chdir="$neobert_root"
        --export="ALL,NEOBERT_ROOT=$neobert_root,COMPLEX_ATTENTION_ROOT=$complex_attention_root,DATASET_PATH=$DATASET_PATH"
        "$job_dir/prepare_data.sbatch"
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "FineWeb-Edu preparation:"
        print_command sbatch --parsable "${prep_command[@]}"
        prep_dependency="PREP_JOB_ID"
    else
        prep_dependency="$(submit_job "${prep_command[@]}")"
        echo "FineWeb-Edu preparation: $prep_dependency"
    fi
elif [[ -n "$prep_dependency" ]]; then
    echo "FineWeb-Edu preparation dependency: $prep_dependency"
else
    echo "FineWeb-Edu preparation: skipped (validated $DATASET_PATH)"
fi

previous_job_id="$prep_dependency"
train_job_ids=()
for ((segment_index = 1; segment_index <= TRAIN_SEGMENTS; segment_index++)); do
    dependency_args=()
    if [[ -n "$previous_job_id" ]]; then
        dependency_args=(--dependency="afterok:$previous_job_id")
    fi
    train_command=(
        --partition=slowlane
        --gpus=A100:1
        --qos=hiwi_project
        --chdir="$neobert_root"
        "${dependency_args[@]}"
        --export="ALL,NEOBERT_ROOT=$neobert_root,COMPLEX_ATTENTION_ROOT=$complex_attention_root,DATASET_PATH=$DATASET_PATH,RUNS_ROOT=$RUNS_ROOT,EXPERIMENT_ID=$EXPERIMENT_ID,SEED=$SEED,SEGMENT_INDEX=$segment_index,MAX_STEPS=84000,MICRO_BATCH=4,GRAD_ACCUM=4,WARMUP_STEPS=5469,CHECKPOINT_STEPS=14000,EVAL_STEPS=14000,LOG_INTERVAL=840,MAX_TIME_SECONDS=53640"
        "$job_dir/train.sbatch"
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "Training segment $segment_index:"
        print_command sbatch --parsable "${train_command[@]}"
        previous_job_id="TRAIN_SEGMENT_${segment_index}_JOB_ID"
        train_job_ids+=("$previous_job_id")
    else
        previous_job_id="$(submit_job "${train_command[@]}")"
        train_job_ids+=("$previous_job_id")
        echo "Training segment $segment_index: $previous_job_id"
    fi
done

echo "Experiment id:   $EXPERIMENT_ID"
echo "Experiment root: $RUNS_ROOT/$EXPERIMENT_ID"
echo "Dataset:         $DATASET_PATH"
echo "Training chain:  ${train_job_ids[*]}"
