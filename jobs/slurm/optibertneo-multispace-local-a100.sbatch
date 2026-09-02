#!/usr/bin/env bash
#SBATCH --job-name=optibertneo-multispace-1p3b
#SBATCH --account=hiwi_project_ac_ratan_jonas
#SBATCH --partition=slowlane
#SBATCH --qos=student_high_res
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=A100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

readonly project_root=/mnt/nfs/home/st171793/ComplexAttention/NeoBERT
readonly complex_root=/mnt/nfs/home/st171793/ComplexAttention

module purge
module load CUDA/12.6.0

export OPTIBERT_PROJECT_ROOT=$project_root
export OPTIBERT_PYTHON=/mnt/nfs/home/st171793/.conda/envs/attention_dev/bin/python
export OPTIBERT_DATASET=$project_root/tokenized_datasets/fineweb_edu_roberta_1p6b
export OPTIBERT_VARIANT=multispace
export OPTIBERT_REQUIRED_GPU=a100
export RUN_ROOT=$project_root/logs/optibertneo-local-1p3b/multispace
export HF_HOME=$complex_root/.cache/huggingface
export ACCELERATE_CONFIG=$project_root/conf/accelerate_ddp.yaml
export TORCH_COMPILE=true
export MAX_TIME_SECONDS=171600
export WANDB_MODE=offline
export STAGE_DATASET=0
export SMOKE_TEST=0
export GLOBAL_SEQUENCES=2048
export MICRO_BATCH=8
export DRY_RUN=0
unset MACHINE_RANK

cd "$project_root"
exec bash jobs/slurm/optibertneo-paired-1p3b-2n8g.sbatch
