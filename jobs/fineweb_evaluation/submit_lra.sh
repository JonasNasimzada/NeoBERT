#!/usr/bin/env bash
set -euo pipefail

NEOBERT_ROOT="${NEOBERT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$NEOBERT_ROOT"

preflight_job="$(sbatch --parsable jobs/fineweb_evaluation/lra_preflight.sbatch)"
full_job="$(sbatch --parsable --dependency="afterok:$preflight_job" jobs/fineweb_evaluation/lra.sbatch)"
aggregate_job="$(sbatch --parsable --dependency="afterok:$full_job" jobs/fineweb_evaluation/lra_aggregate.sbatch)"

echo "LRA preflight job: $preflight_job"
echo "LRA full array job: $full_job"
echo "LRA aggregate job: $aggregate_job"
