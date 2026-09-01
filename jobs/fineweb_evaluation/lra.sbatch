#!/usr/bin/env bash
#SBATCH --job-name=ca-lra-cifar
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5%4
#SBATCH --output=ca-lra-cifar-%A_%a.out
#SBATCH --error=ca-lra-cifar-%A_%a.err

set -euo pipefail

NEOBERT_ROOT="${NEOBERT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the NeoBERT repository root}}"
source "$NEOBERT_ROOT/jobs/attention_ablation/common.sh"
export ATTENTION_SPACE=multispace
export ATTENTION_BACKEND=flash
setup_attention_runtime
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

models=(
    "$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model"
    "$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model"
)
variants=(multispace-flash real-flash)
seeds=(42 43 44)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
model_index=$((task_id % 2))
seed_index=$((task_id / 2))
model="${models[$model_index]}"
variant="${variants[$model_index]}"
seed="${seeds[$seed_index]}"

data_root="${LRA_DATA_ROOT:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/lra/cifar10-prepared}"
output_root="${LRA_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/lra_external}"
result="$output_root/cifar10/$variant/seed-$seed/final_results.json"
if [[ -s "$result" && "${FORCE:-0}" != "1" ]]; then
    echo "Skipping completed result: $result"
    exit 0
fi

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "${variant%-flash}" --backend flash --require-a100
"$COMPLEX_ATTN_PYTHON" scripts/fineweb_evaluation/lra.py \
    --model "$model" \
    --variant "$variant" \
    --data-root "$data_root" \
    --output-dir "$output_root" \
    --seed "$seed" \
    --epochs "${LRA_EPOCHS:-5}" \
    --micro-batch-size "${LRA_MICRO_BATCH_SIZE:-4}" \
    --gradient-accumulation "${LRA_GRADIENT_ACCUMULATION:-4}" \
    --eval-batch-size "${LRA_EVAL_BATCH_SIZE:-8}" \
    --backbone-learning-rate "${LRA_BACKBONE_LR:-3e-5}" \
    --head-learning-rate "${LRA_HEAD_LR:-5e-4}" \
    --warmup-epochs "${LRA_WARMUP_EPOCHS:-1}" \
    --weight-decay "${LRA_WEIGHT_DECAY:-0}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}"
