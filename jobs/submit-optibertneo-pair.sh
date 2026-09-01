#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: jobs/submit-optibertneo-pair.sh [real|multispace|both]

Prepare either or both parameter-matched OptiBERTneo 1.3B-token runs as
independent two-node/four-H100-per-node jobs. The default is "both" and a
dry run; live submission requires the two explicit safety settings below.

Required environment:
  OPTIBERT_PYTHON       Executable in the prepared H100 environment.
  H100_PARTITION        Slurm H100 partition (or put --partition in
                        SBATCH_EXTRA_ARGS).
  H100_ACCOUNT          Slurm account (or put --account in
                        SBATCH_EXTRA_ARGS).

Optional environment:
  OPTIBERT_DATASET      Prepared FineWeb-Edu/RoBERTa dataset directory.
  RUNS_ROOT             Parent for stable real/ and multispace/ run roots.
  REAL_RUN_ROOT         Override the real checkpoint/run directory.
  MULTISPACE_RUN_ROOT   Override the multispace checkpoint/run directory.
  MAX_TIME_SECONDS      Coordinated stop time before Slurm timeout (default
                        in the four-hour job: 13200 seconds).
  OPTIBERT_VARIANTS     Selection used when no positional argument is given.
  SBATCH_EXTRA_ARGS     Whitespace-separated site flags such as QoS,
                        reservation, constraint, or time limit.
  DRY_RUN                Defaults to 1: validate inputs and print commands.
                        Set to 0 only for a live full-run submission.
  CONFIRM_FULL_SUBMISSION
                        Must be exactly YES together with DRY_RUN=0.

Safety: unset any inherited SMOKE_TEST, GLOBAL_SEQUENCES, MICRO_BATCH,
NUM_MACHINES, GPUS_PER_NODE, MACHINE_RANK, EXPECTED_NUM_MACHINES,
EXPECTED_WORLD_SIZE, MASTER_ADDR, MASTER_PORT, and ACCELERATE_CONFIG variables.
This helper rejects them and exports the canonical fixed full-run values;
Slurm derives the allocation-specific master address, port, and machine rank.
EOF
    exit 2
}

if (($# > 1)); then
    usage
fi

selection=${1:-${OPTIBERT_VARIANTS:-both}}
case "$selection" in
    real)
        variants=(real)
        ;;
    multispace)
        variants=(multispace)
        ;;
    both|real,multispace|multispace,real)
        variants=(real multispace)
        ;;
    *)
        echo "Variant selection must be real, multispace, or both; got: $selection" >&2
        usage
        ;;
esac

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
job_file=$project_root/jobs/slurm/optibertneo-paired-1p3b-2n8g.sbatch
dataset_path=$(realpath -m "${OPTIBERT_DATASET:-$project_root/tokenized_datasets/fineweb_edu_roberta_1p6b}")
runs_root=$(realpath -m "${RUNS_ROOT:-$project_root/logs/optibertneo-paired-1p3b}")
python_bin=${OPTIBERT_PYTHON:-}
dry_run=${DRY_RUN:-1}

if [[ "$dry_run" != 0 && "$dry_run" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1, got: $dry_run" >&2
    exit 2
fi

unsafe_inherited_variables=(
    SMOKE_TEST
    GLOBAL_SEQUENCES
    MICRO_BATCH
    NUM_MACHINES
    GPUS_PER_NODE
    MACHINE_RANK
    EXPECTED_NUM_MACHINES
    EXPECTED_WORLD_SIZE
    MASTER_ADDR
    MASTER_PORT
    ACCELERATE_CONFIG
)
for variable_name in "${unsafe_inherited_variables[@]}"; do
    if [[ -v "$variable_name" ]]; then
        echo "Refusing inherited $variable_name; unset it before using the full-run submission helper." >&2
        exit 2
    fi
done

if [[ "$dry_run" == 0 && "${CONFIRM_FULL_SUBMISSION:-}" != YES ]]; then
    echo "Live submission requires DRY_RUN=0 and CONFIRM_FULL_SUBMISSION=YES." >&2
    exit 2
fi
if [[ ! -f "$job_file" ]]; then
    echo "Paired Slurm job is missing: $job_file" >&2
    exit 1
fi
if [[ -z "$python_bin" ]]; then
    echo "OPTIBERT_PYTHON must name the executable in the prepared H100 environment." >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "OPTIBERT_PYTHON is not executable: $python_bin" >&2
    exit 2
fi
for required_dataset_file in \
    "$dataset_path/dataset_info.json" \
    "$dataset_path/optibertneo_manifest.json" \
    "$dataset_path/tokenizer/tokenizer.json"; do
    if [[ ! -f "$required_dataset_file" ]]; then
        echo "Prepared OptiBERTneo dataset is incomplete; missing: $required_dataset_file" >&2
        echo "Run jobs/slurm/prepare-optibertneo-data.sbatch before submission." >&2
        exit 2
    fi
done
if [[ "$dry_run" == 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; use a Slurm login node or set DRY_RUN=1." >&2
    exit 2
fi

read -r -a sbatch_extra_args <<<"${SBATCH_EXTRA_ARGS:-}"

has_option() {
    local long_name=$1
    local short_name=$2
    local argument
    for argument in "${sbatch_extra_args[@]}"; do
        if [[ "$argument" == "$long_name" \
            || "$argument" == "$long_name="* \
            || "$argument" == "$short_name" \
            || "$argument" == "$short_name"* ]]; then
            return 0
        fi
    done
    return 1
}

site_args=()
if [[ -n "${H100_PARTITION:-}" ]]; then
    if has_option --partition -p; then
        echo "Specify the H100 partition with either H100_PARTITION or SBATCH_EXTRA_ARGS, not both." >&2
        exit 2
    fi
    site_args+=(--partition="$H100_PARTITION")
elif ! has_option --partition -p; then
    echo "Set H100_PARTITION or include --partition in SBATCH_EXTRA_ARGS." >&2
    exit 2
fi

if [[ -n "${H100_ACCOUNT:-}" ]]; then
    if has_option --account -A; then
        echo "Specify the Slurm account with either H100_ACCOUNT or SBATCH_EXTRA_ARGS, not both." >&2
        exit 2
    fi
    site_args+=(--account="$H100_ACCOUNT")
elif ! has_option --account -A; then
    echo "Set H100_ACCOUNT or include --account in SBATCH_EXTRA_ARGS." >&2
    exit 2
fi

for exported_value in "$project_root" "$dataset_path" "$runs_root" "$python_bin"; do
    if [[ "$exported_value" == *","* || "$exported_value" == *$'\n'* ]]; then
        echo "Slurm-exported values may not contain commas or newlines: $exported_value" >&2
        exit 2
    fi
done

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

submit_job() {
    local result
    result=$(sbatch --parsable "$@")
    result=${result%%;*}
    if [[ ! "$result" =~ ^[0-9]+$ ]]; then
        echo "sbatch returned an invalid job id: $result" >&2
        return 2
    fi
    printf '%s\n' "$result"
}

for variant in "${variants[@]}"; do
    case "$variant" in
        real)
            run_root=$(realpath -m "${REAL_RUN_ROOT:-$runs_root/real}")
            micro_batch=32
            ;;
        multispace)
            run_root=$(realpath -m "${MULTISPACE_RUN_ROOT:-$runs_root/multispace}")
            micro_batch=8
            ;;
    esac
    if [[ "$run_root" == *","* || "$run_root" == *$'\n'* ]]; then
        echo "Slurm-exported RUN_ROOT may not contain commas or newlines: $run_root" >&2
        exit 2
    fi

    command_args=(
        "${site_args[@]}"
        "${sbatch_extra_args[@]}"
        --job-name="optibertneo-$variant-1p3b"
        --chdir="$project_root"
        --export="ALL,DRY_RUN=0,SMOKE_TEST=0,GLOBAL_SEQUENCES=2048,NUM_MACHINES=2,GPUS_PER_NODE=4,EXPECTED_NUM_MACHINES=2,EXPECTED_WORLD_SIZE=8,MICRO_BATCH=$micro_batch,ACCELERATE_CONFIG=$project_root/conf/accelerate_ddp.yaml,OPTIBERT_PROJECT_ROOT=$project_root,OPTIBERT_PYTHON=$python_bin,OPTIBERT_DATASET=$dataset_path,OPTIBERT_VARIANT=$variant,RUN_ROOT=$run_root"
        "$job_file"
    )

    if [[ "$dry_run" == 1 ]]; then
        echo "OptiBERTneo $variant:"
        print_command sbatch --parsable "${command_args[@]}"
    else
        job_id=$(submit_job "${command_args[@]}")
        echo "OptiBERTneo $variant: job $job_id"
    fi
    echo "  run_root=$run_root"
done
