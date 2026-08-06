#!/usr/bin/env bash
#SBATCH --job-name=ca-babylm-eval
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=ca-babylm-eval-%A_%a.out
#SBATCH --error=ca-babylm-eval-%A_%a.err

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

DATASET_PATH="$(realpath "${DATASET_PATH:?Set DATASET_PATH to the prepared BabyLM DatasetDict directory.}")"
RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$NEOBERT_ROOT/logs/attention_ablation}")"
SEED="${SEED:-42}"
RUN_ROOT="$RUNS_ROOT/$ATTENTION_VARIANT/seed-$SEED"
MODEL_PATH="$RUN_ROOT/final_model"
BENCHMARK_ROOT="$RUN_ROOT/benchmarks"
WANDB_GROUP="complex-attention-babylm-2026-seed-$SEED"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Completed final export is missing: $MODEL_PATH/config.json" >&2
    exit 2
fi

mkdir -p "$BENCHMARK_ROOT" "${WANDB_DIR:-$RUNS_ROOT/wandb}"
export WANDB_DIR="${WANDB_DIR:-$RUNS_ROOT/wandb}"
cd "$NEOBERT_ROOT"

"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/benchmark_mlm.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output "$BENCHMARK_ROOT/heldout_mlm.json" \
    --variant "$ATTENTION_VARIANT" \
    --seed "$SEED" \
    --project "${WANDB_PROJECT:-complex-attention-ablation}" \
    --entity "${WANDB_ENTITY:-}" \
    --group "$WANDB_GROUP" \
    --mode "${WANDB_MODE:-online}"

# Optional official suite. Clone/pin babylm-org/babylm-eval and prepare its
# gated evaluation data outside this job, then export BABYLM_EVAL_ROOT.
if [[ -n "${BABYLM_EVAL_ROOT:-}" ]]; then
    if [[ -f "$BABYLM_EVAL_ROOT/strict/scripts/eval_zero_shot.sh" ]]; then
        eval_script="$BABYLM_EVAL_ROOT/strict/scripts/eval_zero_shot.sh"
        eval_workdir="$BABYLM_EVAL_ROOT/strict"
    elif [[ -f "$BABYLM_EVAL_ROOT/eval_zero_shot.sh" ]]; then
        eval_script="$BABYLM_EVAL_ROOT/eval_zero_shot.sh"
        eval_workdir="$BABYLM_EVAL_ROOT"
    else
        echo "Could not find an executable BabyLM zero-shot script under BABYLM_EVAL_ROOT." >&2
        exit 2
    fi

    # The official evaluator keys its output by the model path's basename.
    # Every training export is named final_model, so give each array task a
    # unique, scoped symlink to avoid nine concurrent jobs colliding.
    official_model_name="hf-$ATTENTION_VARIANT-seed-$SEED"
    official_model_path="$BENCHMARK_ROOT/$official_model_name"
    if [[ -e "$official_model_path" && ! -L "$official_model_path" ]]; then
        echo "Refusing to replace non-symlink official model path: $official_model_path" >&2
        exit 2
    fi
    if [[ ! -L "$official_model_path" ]]; then
        ln -s "$MODEL_PATH" "$official_model_path"
    fi

    (
        cd "$eval_workdir"
        export NEOBERT_BABYLM_FORCE_BF16=1
        export PYTHONPATH="$NEOBERT_ROOT/scripts/attention_ablation/babylm_compat:$PYTHONPATH"
        bash -e "$eval_script" "$official_model_path" mlm "${BABYLM_EVAL_DATA:-evaluation_data/full_eval}"
    )

    default_results_root="$eval_workdir/results/$official_model_name"
    BABYLM_RESULTS_ROOT="${BABYLM_RESULTS_ROOT:-$default_results_root}"
    "$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/log_babylm_results.py \
        --results "$BABYLM_RESULTS_ROOT" \
        --variant "$ATTENTION_VARIANT" \
        --seed "$SEED" \
        --project "${WANDB_PROJECT:-complex-attention-ablation}" \
        --entity "${WANDB_ENTITY:-}" \
        --group "$WANDB_GROUP" \
        --mode "${WANDB_MODE:-online}"
else
    echo "BABYLM_EVAL_ROOT is unset; held-out MLM benchmark completed, official BabyLM suite skipped."
fi
