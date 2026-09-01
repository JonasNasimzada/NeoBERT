#!/usr/bin/env bash

# Complete the official BabyLM tasks that were added after the core zero-shot
# suite.  Each family is failure-isolated so Reading can never prevent both
# GlobalPIQA evaluations from running (or vice versa).

set -uo pipefail

model_path="${1:?Pass the local Hugging Face model path.}"
eval_dir="${2:?Pass the BabyLM full-evaluation data directory.}"
results_root="${3:-results}"
python_bin="${COMPLEX_ATTN_PYTHON:-python}"
expected_scipy="${BABYLM_EXPECTED_SCIPY_VERSION:-1.15.2}"

required_inputs=(
    "$model_path/config.json"
    "$eval_dir/reading/reading_data.csv"
    "$eval_dir/global_piqa_parallel/eng_latn.jsonl"
    "$eval_dir/global_piqa_nonparallel/eng_latn.jsonl"
)
for required_input in "${required_inputs[@]}"; do
    if [[ ! -s "$required_input" ]]; then
        echo "BabyLM completion input is missing or empty: $required_input" >&2
        exit 2
    fi
done

"$python_bin" - "$expected_scipy" <<'PY'
import sys

import scipy
import statsmodels
import statsmodels.formula.api as smf
import torch
from scipy._lib._util import _lazywhere

expected_scipy = sys.argv[1]
if scipy.__version__ != expected_scipy:
    raise RuntimeError(
        f"BabyLM requires scipy=={expected_scipy}; loaded {scipy.__version__} "
        f"from {scipy.__file__}"
    )
if not callable(smf.ols):
    raise RuntimeError("statsmodels.formula.api.ols is unavailable")
if not torch.cuda.is_available():
    raise RuntimeError("official BabyLM completion evaluation requires CUDA")
print(
    "BabyLM completion runtime:",
    f"device={torch.cuda.get_device_name()}",
    f"scipy={scipy.__version__}",
    f"statsmodels={statsmodels.__version__}",
)
PY
preflight_status=$?
if (( preflight_status != 0 )); then
    exit "$preflight_status"
fi

model_name="$(basename "$model_path")"
task_root="$results_root/$model_name/main/zero_shot/mlm"
force_rerun="${BABYLM_FORCE_RERUN:-0}"
failed=0

run_global_piqa() {
    local task="$1"
    local report="$task_root/$task/$task/best_temperature_report.txt"
    local predictions="$task_root/$task/$task/predictions.json"
    if [[ "$force_rerun" != "1" && -s "$report" && -s "$predictions" ]]; then
        echo "Skipping completed BabyLM task: $task"
        return 0
    fi
    echo "Running BabyLM task: $task"
    if "$python_bin" -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$model_path" \
        --backend mlm \
        --task "$task" \
        --data_path "$eval_dir/$task" \
        --output_dir "$results_root" \
        --save_predictions; then
        if [[ ! -s "$report" || ! -s "$predictions" ]]; then
            echo "BabyLM task did not create its required outputs: $task" >&2
            return 1
        fi
        return 0
    fi
    return 1
}

run_reading() {
    local report="$task_root/reading/report.txt"
    local predictions="$task_root/reading/predictions.json"
    if [[ "$force_rerun" != "1" && -s "$report" && -s "$predictions" ]]; then
        echo "Skipping completed BabyLM task: reading"
        return 0
    fi
    echo "Running BabyLM task: reading"
    if "$python_bin" -m evaluation_pipeline.reading.run \
        --model_path_or_name "$model_path" \
        --backend mlm \
        --data_path "$eval_dir/reading/reading_data.csv" \
        --output_dir "$results_root" \
        --revision_name main; then
        if [[ ! -s "$report" || ! -s "$predictions" ]]; then
            echo "BabyLM task did not create its required outputs: reading" >&2
            return 1
        fi
        return 0
    fi
    return 1
}

for task in global_piqa_parallel global_piqa_nonparallel; do
    if ! run_global_piqa "$task"; then
        echo "BabyLM task failed: $task" >&2
        failed=1
    fi
done
if ! run_reading; then
    echo "BabyLM task failed: reading" >&2
    failed=1
fi

exit "$failed"
