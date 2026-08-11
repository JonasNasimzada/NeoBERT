#!/usr/bin/env bash
#SBATCH --job-name=ca-attn-papers
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=ca-attn-papers-%A_%a.out
#SBATCH --error=ca-attn-papers-%A_%a.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/attention_ablation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/attention_ablation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "$job_dir/common.sh" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    echo "Submit from the NeoBERT root or export NEOBERT_ROOT explicitly." >&2
    exit 2
fi
source "$job_dir/common.sh"
resolve_attention_variant "${SLURM_ARRAY_TASK_ID:?This job must run as an array task.}"
setup_attention_runtime

RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$NEOBERT_ROOT/logs/attention_ablation}")"
EXPERIMENT_ID="${EXPERIMENT_ID:-a100-1p376b-v1}"
SEED="${SEED:-42}"
validate_experiment_id "$EXPERIMENT_ID"
validate_attention_seed "$SEED"
RUNS_ROOT="$RUNS_ROOT/$EXPERIMENT_ID"
RUN_ROOT="$RUNS_ROOT/$ATTENTION_VARIANT/seed-$SEED"
BENCHMARK_ROOT="$RUN_ROOT/benchmarks"
WANDB_RUN_GROUP="complex-attention-babylm-2026-$EXPERIMENT_ID-seed-$SEED"
WANDB_RUN_ID="ca-${EXPERIMENT_ID}-${ATTENTION_VARIANT}-s${SEED}-attention-papers"

mkdir -p "$BENCHMARK_ROOT" "${WANDB_DIR:-$RUNS_ROOT/wandb}"
export WANDB_DIR="${WANDB_DIR:-$RUNS_ROOT/wandb}"

cd "$NEOBERT_ROOT"
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space "$ATTENTION_SPACE" \
    --backend "$ATTENTION_BACKEND" \
    --require-a100

"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/benchmark_attention_kernels.py \
    --protocol "${ATTENTION_PAPER_PROTOCOL:-both}" \
    --variants "$ATTENTION_VARIANT" \
    --seed "$SEED" \
    --output "$BENCHMARK_ROOT/attention_papers.json" \
    --wandb-mode "${WANDB_MODE:-online}" \
    --wandb-project "${WANDB_PROJECT:-complex-attention-ablation}" \
    --wandb-entity "${WANDB_ENTITY:-hyper_attention}" \
    --wandb-group "$WANDB_RUN_GROUP" \
    --wandb-name "$EXPERIMENT_ID-$ATTENTION_VARIANT-attention-papers" \
    --wandb-id "$WANDB_RUN_ID"
