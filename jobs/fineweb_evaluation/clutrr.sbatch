#!/usr/bin/env bash
#SBATCH --job-name=ca-clutrr
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=ca-clutrr-%A_%a.out
#SBATCH --error=ca-clutrr-%A_%a.err

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

data_root="${CLUTRR_DATA_ROOT:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/data}"
output_root="${EVALUATION_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/clutrr}"
configs="${CLUTRR_CONFIGS:-gen_train23_test2to10 rob_train_clean_23_test_all_23 rob_train_disc_23_test_all_23 rob_train_irr_23_test_all_23 rob_train_sup_23_test_all_23}"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "${variant%-flash}" --backend flash --require-a100

for config in $configs; do
    result="$output_root/$config/$variant/seed-$seed/final_results.json"
    if [[ -s "$result" && "${FORCE:-0}" != "1" ]]; then
        echo "Skipping completed result: $result"
        continue
    fi
    "$COMPLEX_ATTN_PYTHON" scripts/fineweb_evaluation/clutrr.py \
        --model "$model" \
        --variant "$variant" \
        --data-root "$data_root" \
        --config "$config" \
        --output-dir "$output_root" \
        --seed "$seed" \
        --epochs "${CLUTRR_EPOCHS:-10}" \
        --batch-size "${CLUTRR_BATCH_SIZE:-32}" \
        --eval-batch-size "${CLUTRR_EVAL_BATCH_SIZE:-64}" \
        --max-length "${CLUTRR_MAX_LENGTH:-512}"
done
