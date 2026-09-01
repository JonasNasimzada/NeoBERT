#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
finetune_job="${1:-67526}"
if [[ ! "$finetune_job" =~ ^[0-9]+$ ]]; then
    echo "Fine-tune job ID must be numeric; got $finetune_job" >&2
    exit 2
fi

hans_finetune_dependency="afterok:${finetune_job}_6:${finetune_job}_13"
paws_finetune_dependency="afterok:${finetune_job}_5:${finetune_job}_12"

hans_preflight="$(
    sbatch --parsable \
        --dependency="$hans_finetune_dependency" \
        --export=ALL,NEOBERT_ROOT="$repo_root",EVAL_TASK=hans \
        "$repo_root/jobs/fineweb_evaluation/hans_paws_preflight.sbatch"
)"
paws_preflight="$(
    sbatch --parsable \
        --dependency="$paws_finetune_dependency" \
        --export=ALL,NEOBERT_ROOT="$repo_root",EVAL_TASK=paws \
        "$repo_root/jobs/fineweb_evaluation/hans_paws_preflight.sbatch"
)"
hans_full="$(
    sbatch --parsable \
        --dependency="afterok:$hans_preflight" \
        --export=ALL,NEOBERT_ROOT="$repo_root",EVAL_TASK=hans \
        "$repo_root/jobs/fineweb_evaluation/hans_paws.sbatch"
)"
paws_full="$(
    sbatch --parsable \
        --dependency="afterok:$paws_preflight" \
        --export=ALL,NEOBERT_ROOT="$repo_root",EVAL_TASK=paws \
        "$repo_root/jobs/fineweb_evaluation/hans_paws.sbatch"
)"

printf '{"finetune_job":"%s","hans_preflight":"%s","paws_preflight":"%s","hans_full":"%s","paws_full":"%s"}\n' \
    "$finetune_job" "$hans_preflight" "$paws_preflight" "$hans_full" "$paws_full"
