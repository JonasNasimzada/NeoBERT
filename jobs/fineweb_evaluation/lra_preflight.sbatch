#!/usr/bin/env bash
#SBATCH --job-name=ca-lra-preflight
#SBATCH --partition=fastlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=ca-lra-preflight-%j.out
#SBATCH --error=ca-lra-preflight-%j.err

set -euo pipefail

NEOBERT_ROOT="${NEOBERT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the NeoBERT repository root}}"
source "$NEOBERT_ROOT/jobs/attention_ablation/common.sh"
export ATTENTION_SPACE=multispace
export ATTENTION_BACKEND=flash
setup_attention_runtime
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

data_root="${LRA_DATA_ROOT:-$COMPLEX_ATTENTION_ROOT/.cache/evaluations/lra/cifar10-prepared}"
output_root="${LRA_PREFLIGHT_OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/preflight-lra-$SLURM_JOB_ID}"
models=(
    "$NEOBERT_ROOT/logs/multispace_fineweb/fineweb-edu-s1024-multispace-100m-v1/multispace-flash/seed-42/final_model"
    "$NEOBERT_ROOT/logs/real_fineweb/fineweb-edu-s1024-real-100m-v1/real-flash/seed-42/final_model"
)
variants=(multispace-flash real-flash)

cd "$NEOBERT_ROOT"
bash -n \
    jobs/fineweb_evaluation/lra_preflight.sbatch \
    jobs/fineweb_evaluation/lra.sbatch \
    jobs/fineweb_evaluation/lra_aggregate.sbatch \
    jobs/fineweb_evaluation/submit_lra.sh
"$COMPLEX_ATTN_PYTHON" -m py_compile \
    scripts/fineweb_evaluation/lra_prepare.py \
    scripts/fineweb_evaluation/lra.py \
    scripts/fineweb_evaluation/lra_aggregate.py

for index in 0 1; do
    "$COMPLEX_ATTN_PYTHON" scripts/attention_ablation/check_runtime.py \
        --space "${variants[$index]%-flash}" --backend flash --require-a100
    "$COMPLEX_ATTN_PYTHON" scripts/fineweb_evaluation/lra.py \
        --model "${models[$index]}" \
        --variant "${variants[$index]}" \
        --data-root "$data_root" \
        --output-dir "$output_root" \
        --seed 42 \
        --epochs 1 \
        --micro-batch-size 4 \
        --gradient-accumulation 2 \
        --eval-batch-size 8 \
        --max-train-examples 32 \
        --max-eval-examples 16 \
        --num-workers 0
done

"$COMPLEX_ATTN_PYTHON" - "$output_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "cifar10"
reports = {
    variant: json.loads(
        (root / variant / "seed-42" / "final_results.json").read_text()
    )
    for variant in ("multispace-flash", "real-flash")
}
left, right = reports.values()
checks = {
    "a100": all("A100" in report["gpu"]["name"] for report in reports.values()),
    "checkpoint_parameters": all(
        report["model"]["checkpoint_parameters"] == 99_985_152
        for report in reports.values()
    ),
    "head_initialization": (
        left["model"]["classifier_head_initialization_sha256"]
        == right["model"]["classifier_head_initialization_sha256"]
    ),
    "data_fingerprint": (
        left["dataset"]["fingerprint_sha256"]
        == right["dataset"]["fingerprint_sha256"]
    ),
    "data_order": left["data_order_sha256"] == right["data_order_sha256"],
    "hyperparameters": left["hyperparameters"] == right["hyperparameters"],
    "sequence_1025": all(
        report["adaptation"]["model_sequence_length"] == 1025
        for report in reports.values()
    ),
    "no_truncation": all(
        report["adaptation"]["truncation"] == "none"
        for report in reports.values()
    ),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"paired LRA preflight failed: {failed}")
print(json.dumps({"checks": checks, "head_sha256": left["model"]["classifier_head_initialization_sha256"]}, indent=2))
PY
