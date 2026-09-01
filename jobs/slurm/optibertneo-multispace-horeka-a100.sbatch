#!/usr/bin/env bash
#SBATCH --job-name=optibertneo-ms-1p3b
#SBATCH --account=hk-project-pai00130
#SBATCH --partition=accelerated
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Legacy HoreKa Green is scheduled to close after 2026-09-15. Migrating this
# job to HoreKa 2 also requires migrating its /hkfs code, data, and environment.
export MAMBA_ENV=/hkfs/home/project/hk-project-pai00051/st_st171793/envs/attention_dev
export MAMBA_EXE=/hkfs/home/project/hk-project-pai00051/st_st171793/micromamba/micromamba
export MAMBA_ROOT_PREFIX=/hkfs/home/project/hk-project-pai00051/st_st171793

module purge
module load compiler/gnu/13 devel/cuda/12.9
module list
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate "$MAMBA_ENV"

complex_root=/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention
export OPTIBERT_PROJECT_ROOT="$complex_root/NeoBERT"
export OPTIBERT_PYTHON="$MAMBA_ENV/bin/python"
export OPTIBERT_VARIANT=multispace
export OPTIBERT_REQUIRED_GPU=a100
export OPTIBERT_DATASET=${OPTIBERT_DATASET:-$OPTIBERT_PROJECT_ROOT/tokenized_datasets/fineweb_edu_roberta_1p6b}
export RUN_ROOT=${RUN_ROOT:-$OPTIBERT_PROJECT_ROOT/logs/optibertneo-horeka-1p3b/multispace}
export ACCELERATE_CONFIG=$OPTIBERT_PROJECT_ROOT/conf/accelerate_ddp.yaml
export TORCH_COMPILE=true MAX_TIME_SECONDS=171600
export SLURM_TMPDIR=${TMPDIR:?HoreKa did not provide a node-local TMPDIR}
export STAGE_DATASET=${STAGE_DATASET:-1}
export SMOKE_TEST=0 GLOBAL_SEQUENCES=2048 MICRO_BATCH=8 DRY_RUN=0
unset MACHINE_RANK

cd "$OPTIBERT_PROJECT_ROOT"
exec bash jobs/slurm/optibertneo-paired-1p3b-2n8g.sbatch
