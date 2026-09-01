#!/usr/bin/env bash

# Run the complete public BabyLM 2026 Strict final-checkpoint diagnostics.
# FineWeb-Edu models are external diagnostics and are not leaderboard eligible.

set -uo pipefail

model_path="${1:?Pass a uniquely named local Hugging Face model path.}"
eval_dir="${2:?Pass the BabyLM full-evaluation data directory.}"
results_root="${3:?Pass the dedicated result root.}"
python_bin="${COMPLEX_ATTN_PYTHON:-python}"
force_rerun="${BABYLM_FORCE_RERUN:-0}"
batch_size="${BABYLM_BATCH_SIZE:-32}"
non_causal_batch_size="${BABYLM_NON_CAUSAL_BATCH_SIZE:-32}"

required_inputs=(
    "$model_path/config.json"
    "$model_path/model.safetensors"
    "$eval_dir/reading/reading_data.csv"
    "$eval_dir/global_piqa_parallel/eng_latn.jsonl"
    "$eval_dir/global_piqa_nonparallel/eng_latn.jsonl"
)
for required_input in "${required_inputs[@]}"; do
    if [[ ! -s "$required_input" ]]; then
        echo "BabyLM evaluation input is missing or empty: $required_input" >&2
        exit 2
    fi
done
for data_dir in blimp_filtered supplement_filtered ewok_filtered entity_tracking comps; do
    if [[ ! -d "$eval_dir/$data_dir" ]] \
        || [[ -z "$(find "$eval_dir/$data_dir" -type f -size +0c -print -quit)" ]]; then
        echo "BabyLM evaluation directory is missing or empty: $eval_dir/$data_dir" >&2
        exit 2
    fi
done

model_name="$(basename "$model_path")"
task_root="$results_root/$model_name/main/zero_shot/mlm"
mkdir -p "$results_root"
failed=0

run_ranked_task() {
    local task="$1"
    local dataset="$2"
    local report="$task_root/$task/$dataset/best_temperature_report.txt"
    local predictions="$task_root/$task/$dataset/predictions.json"
    if [[ "$force_rerun" != "1" && -s "$report" && -s "$predictions" ]]; then
        echo "Skipping completed BabyLM task: $task/$dataset"
        return 0
    fi
    echo "Running BabyLM task: $task/$dataset"
    if "$python_bin" -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$model_path" \
        --backend mlm \
        --task "$task" \
        --data_path "$eval_dir/$dataset" \
        --output_dir "$results_root" \
        --batch_size "$batch_size" \
        --non_causal_batch_size "$non_causal_batch_size" \
        --save_predictions; then
        if [[ ! -s "$report" || ! -s "$predictions" ]]; then
            echo "BabyLM task did not create required outputs: $task/$dataset" >&2
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
        --output_dir "$results_root"; then
        if [[ ! -s "$report" || ! -s "$predictions" ]]; then
            echo "BabyLM task did not create required outputs: reading" >&2
            return 1
        fi
        return 0
    fi
    return 1
}

ranked_tasks=(
    "blimp:blimp_filtered"
    "blimp:supplement_filtered"
    "ewok:ewok_filtered"
    "entity_tracking:entity_tracking"
    "comps:comps"
    "global_piqa_parallel:global_piqa_parallel"
    "global_piqa_nonparallel:global_piqa_nonparallel"
)
for task_spec in "${ranked_tasks[@]}"; do
    task="${task_spec%%:*}"
    dataset="${task_spec#*:}"
    if ! run_ranked_task "$task" "$dataset"; then
        echo "BabyLM task failed: $task/$dataset" >&2
        failed=1
    fi
done
if ! run_reading; then
    echo "BabyLM task failed: reading" >&2
    failed=1
fi

exit "$failed"
