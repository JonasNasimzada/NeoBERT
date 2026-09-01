#!/usr/bin/env bash
#SBATCH --job-name=ca-babylm-completion
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=ca-babylm-completion-%A_%a.out
#SBATCH --error=ca-babylm-completion-%A_%a.err

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
    exit 2
fi
source "$job_dir/common.sh"

# Keep the historical fused checkpoint in this completion-only matrix.  It is
# no longer a distinct new-training variant, but it remains part of this run's
# twelve-checkpoint result set.
completion_variants=(
    complex-native
    complex-torch
    complex-flash
    split-native
    split-torch
    real-torch
    real-flash
    split-flash
    dual-native
    dual-torch
    dual-flash
    dual-flash-fused
)
completion_spaces=(complex complex complex split split real real split dual dual dual dual)
completion_backends=(native torch flash native torch torch flash flash native torch flash flash_fused)

task_id="${SLURM_ARRAY_TASK_ID:?This job must run as an array task.}"
if [[ ! "$task_id" =~ ^[0-9]+$ ]] \
    || (( 10#$task_id >= ${#completion_variants[@]} )); then
    echo "Array task id must be an integer from 0 through 11; got '$task_id'." >&2
    exit 2
fi
ATTENTION_VARIANT="${completion_variants[$task_id]}"
ATTENTION_SPACE="${completion_spaces[$task_id]}"
ATTENTION_BACKEND="${completion_backends[$task_id]}"
export ATTENTION_VARIANT ATTENTION_SPACE ATTENTION_BACKEND
setup_attention_runtime

RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$NEOBERT_ROOT/logs/attention_ablation}")"
EXPERIMENT_ID="${EXPERIMENT_ID:-babylm-full-dflash-s1024-20260821-v1}"
SEED="${SEED:-42}"
validate_experiment_id "$EXPERIMENT_ID"
validate_attention_seed "$SEED"
RUN_ROOT="$RUNS_ROOT/$EXPERIMENT_ID/$ATTENTION_VARIANT/seed-$SEED"
MODEL_PATH="$RUN_ROOT/final_model"
BENCHMARK_ROOT="$RUN_ROOT/benchmarks"
WANDB_GROUP="complex-attention-babylm-2026-$EXPERIMENT_ID-seed-$SEED"

if [[ ! -s "$MODEL_PATH/config.json" ]]; then
    echo "Completed final export is missing: $MODEL_PATH/config.json" >&2
    exit 2
fi
BABYLM_EVAL_ROOT="$(realpath "${BABYLM_EVAL_ROOT:?Set BABYLM_EVAL_ROOT to the pinned evaluator checkout.}")"
eval_workdir="$BABYLM_EVAL_ROOT/strict"
eval_data="${BABYLM_EVAL_DATA:-$eval_workdir/evaluation_data/full_eval}"
eval_deps="${BABYLM_EVAL_DEPS:-$BABYLM_EVAL_ROOT/.eval_deps}"
if [[ ! -d "$eval_deps" ]]; then
    echo "BabyLM evaluator dependency overlay is missing: $eval_deps" >&2
    exit 2
fi

mkdir -p "$BENCHMARK_ROOT" "${WANDB_DIR:-$RUNS_ROOT/$EXPERIMENT_ID/wandb}"
export WANDB_DIR="${WANDB_DIR:-$RUNS_ROOT/$EXPERIMENT_ID/wandb}"
official_model_name="hf-$EXPERIMENT_ID-$ATTENTION_VARIANT-seed-$SEED"
official_model_path="$BENCHMARK_ROOT/$official_model_name"
if [[ -e "$official_model_path" && ! -L "$official_model_path" ]]; then
    echo "Refusing to replace non-symlink official model path: $official_model_path" >&2
    exit 2
fi
if [[ ! -L "$official_model_path" ]]; then
    ln -s "$MODEL_PATH" "$official_model_path"
fi

resource_report="$BENCHMARK_ROOT/babylm_completion_resource_usage.json"
results_root="${BABYLM_RESULTS_BASE:-$eval_workdir/results}"
export NEOBERT_BABYLM_FORCE_BF16=1
export NEOBERT_BABYLM_SEED="$SEED"
export PYTHONPATH="$NEOBERT_ROOT/scripts/attention_ablation/babylm_compat:$eval_deps:$PYTHONPATH"

cd "$eval_workdir"
task_root="$results_root/$official_model_name/main/zero_shot/mlm"
required_outputs=(
    "$task_root/reading/report.txt"
    "$task_root/reading/predictions.json"
    "$task_root/global_piqa_parallel/global_piqa_parallel/best_temperature_report.txt"
    "$task_root/global_piqa_parallel/global_piqa_parallel/predictions.json"
    "$task_root/global_piqa_nonparallel/global_piqa_nonparallel/best_temperature_report.txt"
    "$task_root/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json"
)
completion_ready=1
for required_output in "${required_outputs[@]}"; do
    if [[ ! -s "$required_output" ]]; then
        completion_ready=0
        break
    fi
done
resource_ready=0
if [[ -s "$resource_report" ]] && "$COMPLEX_ATTN_PYTHON" - "$resource_report" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if report.get("exit_code") == 0 else 1)
PY
then
    resource_ready=1
fi

if [[ "${BABYLM_FORCE_RERUN:-0}" != "1" ]] \
    && (( completion_ready == 1 && resource_ready == 1 )); then
    echo "All BabyLM completion outputs already exist; preserving the successful resource report."
else
    "$COMPLEX_ATTN_PYTHON" \
        "$NEOBERT_ROOT/scripts/attention_ablation/run_with_resource_monitor.py" \
        --output "$resource_report" \
        --sample-interval "${BENCHMARK_MEMORY_SAMPLE_INTERVAL:-0.25}" \
        -- bash \
        "$NEOBERT_ROOT/scripts/attention_ablation/run_babylm_completion.sh" \
        "$official_model_path" "$eval_data" "$results_root"
fi

model_results="$results_root/$official_model_name"
"$COMPLEX_ATTN_PYTHON" "$NEOBERT_ROOT/scripts/attention_ablation/log_babylm_results.py" \
    --results "$model_results" \
    --resource-report "$resource_report" \
    --variant "$ATTENTION_VARIANT" \
    --seed "$SEED" \
    --id-prefix "$EXPERIMENT_ID" \
    --project "${WANDB_PROJECT:-complex-attention-ablation}" \
    --entity "${WANDB_ENTITY:-hyper_attention}" \
    --group "$WANDB_GROUP" \
    --mode "${WANDB_MODE:-online}"
