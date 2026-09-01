#!/usr/bin/env bash
set -euo pipefail

: "${OPTIBERT_PROJECT_ROOT:?OPTIBERT_PROJECT_ROOT must be exported}"
: "${OPTIBERT_REQUIRED_GPU:?OPTIBERT_REQUIRED_GPU must be exported}"
: "${RUN_ROOT:?RUN_ROOT must be exported}"

export MAMBA_ROOT_PREFIX=/hkfs/home/project/hk-project-pai00051/st_st171793
export MAMBA_ENV=$MAMBA_ROOT_PREFIX/envs/attention_dev
export MAMBA_EXE=$MAMBA_ROOT_PREFIX/micromamba/micromamba

module purge
module load compiler/gnu/13
module load devel/cuda/12.9
module list
nvcc --version
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate "$MAMBA_ENV"

export OPTIBERT_PYTHON=$MAMBA_ENV/bin/python
export OPTIBERT_VARIANT=multispace
export OPTIBERT_DATASET=${OPTIBERT_DATASET:-$OPTIBERT_PROJECT_ROOT/tokenized_datasets/fineweb_edu_roberta_1p6b}
export ACCELERATE_CONFIG=$OPTIBERT_PROJECT_ROOT/conf/accelerate_ddp.yaml
export TORCH_COMPILE=true MAX_TIME_SECONDS=171600
export SLURM_TMPDIR=${TMPDIR:?HoreKa did not provide a node-local TMPDIR}
export STAGE_DATASET=${STAGE_DATASET:-1}
export SMOKE_TEST=0 GLOBAL_SEQUENCES=2048 MICRO_BATCH=8 DRY_RUN=0
unset MACHINE_RANK

cd "$OPTIBERT_PROJECT_ROOT"
exec bash jobs/slurm/optibertneo-paired-1p3b-2n8g.sbatch
