#!/usr/bin/env bash
#SBATCH --job-name=ca-clutrr-preflight
#SBATCH --partition=fastlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=ca-clutrr-preflight-%j.out
#SBATCH --error=ca-clutrr-preflight-%j.err

set -euo pipefail

NEOBERT_ROOT="${NEOBERT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the NeoBERT repository root}}"
source "$NEOBERT_ROOT/jobs/attention_ablation/common.sh"
export ATTENTION_SPACE=multispace
export ATTENTION_BACKEND=flash
setup_attention_runtime
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

data_root="${CLUTRR_DATA_ROOT:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/data}"
output_root="${EVALUATION_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/preflight-clutrr-$SLURM_JOB_ID}"
models=(
    "$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model"
    "$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model"
)
variants=(multispace-flash real-flash)

cd "$NEOBERT_ROOT"
bash -n jobs/fineweb_evaluation/clutrr.sbatch jobs/fineweb_evaluation/clutrr_preflight.sbatch
"$COMPLEX_ATTN_PYTHON" -m py_compile scripts/fineweb_evaluation/clutrr.py

for index in 0 1; do
    "$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
        --space "${variants[$index]%-flash}" --backend flash --require-a100
    "$COMPLEX_ATTN_PYTHON" scripts/fineweb_evaluation/clutrr.py \
        --model "${models[$index]}" \
        --variant "${variants[$index]}" \
        --data-root "$data_root" \
        --config gen_train23_test2to10 \
        --output-dir "$output_root" \
        --seed 42 \
        --epochs 1 \
        --batch-size 8 \
        --eval-batch-size 8 \
        --max-train-examples 16 \
        --max-eval-examples 8
done

"$COMPLEX_ATTN_PYTHON" - "$output_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "gen_train23_test2to10"
reports = [
    json.loads((root / variant / "seed-42" / "final_results.json").read_text())
    for variant in ("multispace-flash", "real-flash")
]
fingerprints = {
    report["model"]["classifier_head_initialization_sha256"] for report in reports
}
if len(fingerprints) != 1:
    raise RuntimeError(f"paired classifier heads differ: {sorted(fingerprints)}")
print(f"paired_classifier_head_sha256={fingerprints.pop()}")
PY
