#!/usr/bin/env bash

set -euo pipefail

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
neobert_root="$(cd "$job_dir/../.." && pwd)"
: "${DATASET_PATH:?Export DATASET_PATH before submitting the sweep.}"
source "$job_dir/common.sh"
EXPERIMENT_ID="${EXPERIMENT_ID:-a100-3h-v1}"
SEED="${SEED:-42}"
validate_experiment_id "$EXPERIMENT_ID"
validate_attention_seed "$SEED"
RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$neobert_root/logs/attention_ablation}")"
export EXPERIMENT_ID RUNS_ROOT SEED

# A full submission must use the calibrated schedule even if variables from a
# previous two-step smoke command remain in the caller's environment.  Direct
# sbatch submissions can still override these values intentionally.
unset MAX_STEPS WARMUP_STEPS CHECKPOINT_STEPS EVAL_STEPS LOG_INTERVAL MAX_TIME_SECONDS

# Keep the resource request explicit here as well as in the sbatch files so a
# copied submission command retains the requested cluster contract.
train_job_id="$(
    sbatch --parsable \
        --partition=slowlane \
        --gpus=A100:1 \
        --qos=hiwi_project \
        --array=0-8 \
        --chdir="$neobert_root" \
        --export=ALL,NEOBERT_ROOT="$neobert_root" \
        "$job_dir/train.sbatch"
)"
train_job_id="${train_job_id%%;*}"

benchmark_job_id="$(
    sbatch --parsable \
        --partition=slowlane \
        --gpus=A100:1 \
        --qos=hiwi_project \
        --array=0-8 \
        --chdir="$neobert_root" \
        --export=ALL,NEOBERT_ROOT="$neobert_root" \
        --dependency="aftercorr:$train_job_id" \
        "$job_dir/benchmark.sbatch"
)"
benchmark_job_id="${benchmark_job_id%%;*}"

echo "Training array:  $train_job_id"
echo "Benchmark array: $benchmark_job_id (aftercorr:$train_job_id)"
echo "Experiment id:   $EXPERIMENT_ID"
echo "Experiment root: $RUNS_ROOT/$EXPERIMENT_ID"
