#!/usr/bin/env bash
#SBATCH --job-name=ca-trained-mqar-cmp
#SBATCH --partition=slowlane
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=ca-trained-mqar-cmp-%j.out
#SBATCH --error=ca-trained-mqar-cmp-%j.err

set -euo pipefail

if [[ -z "${NEOBERT_ROOT:-}" ]]; then
    NEOBERT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi
NEOBERT_ROOT="$(realpath "$NEOBERT_ROOT")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-$NEOBERT_ROOT/logs/fineweb_evaluation/trained_masked_mqar-v1}")"
cd "$NEOBERT_ROOT"
python3 scripts/attention_ablation/compare_trained_masked_mqar.py \
    --root "$OUTPUT_ROOT" \
    --seeds 42 43 44 \
    --output "$OUTPUT_ROOT/three_seed_comparison.json"
