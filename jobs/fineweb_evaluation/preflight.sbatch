#!/usr/bin/env bash
#SBATCH --job-name=ca-fw-eval-test
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=ca-fw-eval-test-%j.out
#SBATCH --error=ca-fw-eval-test-%j.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/fineweb_evaluation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/fineweb_evaluation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$job_dir/../attention_ablation/common.sh"
export ATTENTION_SPACE=multispace
export ATTENTION_BACKEND=flash
setup_attention_runtime
export PYTHONDONTWRITEBYTECODE=1

DATASET_PATH="$(realpath "${DATASET_PATH:-$NEOBERT_ROOT/tokenized_datasets/fineweb_edu_google_1024_1p6b}")"
MULTISPACE_MODEL_PATH="$(realpath "${MULTISPACE_MODEL_PATH:-$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model}")"
REAL_MODEL_PATH="$(realpath "${REAL_MODEL_PATH:-$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model}")"
PREFLIGHT_ROOT="$(realpath -m "${PREFLIGHT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/preflight-$SLURM_JOB_ID}")"
mkdir -p "$PREFLIGHT_ROOT"

cd "$NEOBERT_ROOT"
bash -n \
    jobs/fineweb_evaluation/preflight.sbatch \
    jobs/fineweb_evaluation/evaluate_mlm.sbatch
"$COMPLEX_ATTN_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_attention_ablation_benchmarks.py::TestVariantMatrix \
    tests/test_fineweb_full_mlm_evaluation.py
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/validate_100m_pair.py
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space multispace --backend flash --require-a100
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
    --space real --backend flash --require-a100

"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/benchmark_mlm.py \
    --model "$MULTISPACE_MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output "$PREFLIGHT_ROOT/multispace-flash.json" \
    --variant multispace-flash \
    --contexts 128 1024 \
    --token-budget 4096 \
    --batch-tokens 4096 \
    --seed 42 \
    --mode disabled
"$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/benchmark_mlm.py \
    --model "$REAL_MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output "$PREFLIGHT_ROOT/real-100m-flash.json" \
    --variant real-100m-flash \
    --contexts 128 1024 \
    --token-budget 4096 \
    --batch-tokens 4096 \
    --seed 42 \
    --mode disabled

"$COMPLEX_ATTN_PYTHON" - "$PREFLIGHT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
reports = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
assert len(reports) == 2
assert {report["variant"] for report in reports} == {
    "multispace-flash",
    "real-100m-flash",
}
assert {report["trainable_parameters"] for report in reports} == {99_985_152}
for report in reports:
    assert set(report["results"]) == {"128", "1024"}
    assert all(item["evaluated_tokens"] == 4096 for item in report["results"].values())
print(f"A100 paired benchmark preflight passed: {root}")
PY
