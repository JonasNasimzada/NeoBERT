#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-hp-pre
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=ca-fw-hp-pre-%j.out

set -euo pipefail
export PREFLIGHT=1

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    eval_job="$NEOBERT_ROOT/jobs/fineweb_evaluation/hans_paws.sbatch"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    eval_job="$SLURM_SUBMIT_DIR/jobs/fineweb_evaluation/hans_paws.sbatch"
else
    eval_job="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hans_paws.sbatch"
fi

# Run the same entry point as the full job while retaining this job's shorter
# preflight allocation and deterministic 32-row, globally-spaced subset.
bash "$eval_job"
