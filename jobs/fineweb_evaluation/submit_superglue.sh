#!/usr/bin/env bash
set -euo pipefail

NEOBERT_ROOT="${NEOBERT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$NEOBERT_ROOT"

preflight="$(sbatch --parsable jobs/fineweb_evaluation/superglue_preflight.sbatch)"
gate="$(sbatch --parsable --dependency="afterok:$preflight" jobs/fineweb_evaluation/superglue_preflight_verify.sbatch)"
full="$(sbatch --parsable --dependency="afterok:$gate" jobs/fineweb_evaluation/superglue.sbatch)"
aggregate="$(sbatch --parsable --dependency="afterok:$full" jobs/fineweb_evaluation/superglue_aggregate.sbatch)"

printf 'preflight=%s\ngate=%s\nfull=%s\naggregate=%s\n' \
    "$preflight" "$gate" "$full" "$aggregate"
