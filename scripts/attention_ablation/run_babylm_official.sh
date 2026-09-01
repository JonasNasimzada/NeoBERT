#!/usr/bin/env bash

# Run the complete BabyLM 2026 Strict full-checkpoint zero-shot suite.

set -euo pipefail

eval_script="${1:?Pass the official eval_zero_shot.sh path.}"
model_path="${2:?Pass the local Hugging Face model path.}"
eval_dir="${3:?Pass the BabyLM full-evaluation data directory.}"
python_bin="${COMPLEX_ATTN_PYTHON:-python}"
expected_scipy="${BABYLM_EXPECTED_SCIPY_VERSION:-1.15.2}"

required_inputs=(
    "$eval_dir/blimp_filtered"
    "$eval_dir/supplement_filtered"
    "$eval_dir/ewok_filtered"
    "$eval_dir/entity_tracking"
    "$eval_dir/comps"
    "$eval_dir/reading/reading_data.csv"
    "$eval_dir/global_piqa_parallel"
    "$eval_dir/global_piqa_nonparallel"
)
for required_input in "${required_inputs[@]}"; do
    if [[ ! -e "$required_input" ]]; then
        echo "BabyLM full-evaluation input is missing: $required_input" >&2
        exit 2
    fi
    if [[ -d "$required_input" ]] \
        && [[ -z "$(find "$required_input" -type f -size +0c -print -quit)" ]]; then
        echo "BabyLM full-evaluation directory has no nonempty files: $required_input" >&2
        exit 2
    fi
    if [[ -f "$required_input" && ! -s "$required_input" ]]; then
        echo "BabyLM full-evaluation input is empty: $required_input" >&2
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
    raise RuntimeError("official BabyLM evaluation requires CUDA")
print(
    "BabyLM official runtime:",
    f"device={torch.cuda.get_device_name()}",
    f"scipy={scipy.__version__}",
    f"statsmodels={statsmodels.__version__}",
)
PY

failed=0
if ! bash -e "$eval_script" "$model_path" mlm "$eval_dir"; then
    echo "The core BabyLM zero-shot suite failed." >&2
    failed=1
fi

# GlobalPIQA was added as a separate script upstream. Evaluate only the final
# checkpoint here; the ablation exports do not contain BabyLM revision branches.
for task in global_piqa_parallel global_piqa_nonparallel; do
    if ! "$python_bin" -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$model_path" \
        --backend mlm \
        --task "$task" \
        --data_path "$eval_dir/$task" \
        --save_predictions; then
        echo "BabyLM task failed: $task" >&2
        failed=1
    fi
done

exit "$failed"
