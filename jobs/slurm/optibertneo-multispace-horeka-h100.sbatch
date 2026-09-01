#!/usr/bin/env bash
#SBATCH --job-name=optibertneo-ms-h100
#SBATCH --account=hk-project-pai00130
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# This targets the legacy HoreKa 1 Teal queue. Check its current availability
# with `sbatch --test-only` before consuming or moving any training state.
export OPTIBERT_PROJECT_ROOT=/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention/NeoBERT
export OPTIBERT_REQUIRED_GPU=h100
export RUN_ROOT=${RUN_ROOT:-$OPTIBERT_PROJECT_ROOT/logs/optibertneo-horeka-h100-1p3b/multispace}
exec "$OPTIBERT_PROJECT_ROOT/jobs/slurm/run-optibertneo-multispace-horeka.sh"
