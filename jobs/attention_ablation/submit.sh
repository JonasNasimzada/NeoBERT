#!/usr/bin/env bash

set -euo pipefail

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_PATH:?Export DATASET_PATH before submitting the sweep.}"

# Keep the resource request explicit here as well as in the sbatch files so a
# copied submission command retains the requested cluster contract.
train_job_id="$(
    sbatch --parsable \
        --partition=slowlane \
        --gpus=A100:1 \
        --qos=hiwi_project \
        --array=0-6 \
        "$job_dir/train.sbatch"
)"
train_job_id="${train_job_id%%;*}"

benchmark_job_id="$(
    sbatch --parsable \
        --partition=slowlane \
        --gpus=A100:1 \
        --qos=hiwi_project \
        --array=0-6 \
        --dependency="aftercorr:$train_job_id" \
        "$job_dir/benchmark.sbatch"
)"
benchmark_job_id="${benchmark_job_id%%;*}"

echo "Training array:  $train_job_id"
echo "Benchmark array: $benchmark_job_id (aftercorr:$train_job_id)"
