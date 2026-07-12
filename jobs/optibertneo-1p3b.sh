#!/usr/bin/env bash
set -euo pipefail

variant=${1:-}
case "$variant" in
    baseline)
        model_config=optibertneo-198m
        default_micro_batch=32
        ;;
    mixed)
        model_config=optibertneo-mixed-198m
        default_micro_batch=4
        ;;
    *)
        echo "Usage: $0 {baseline|mixed}" >&2
        exit 2
        ;;
esac

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
num_machines=${NUM_MACHINES:-${SLURM_JOB_NUM_NODES:-1}}
gpus_per_node=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
machine_rank=${MACHINE_RANK:-${SLURM_NODEID:-0}}
world_size=$((num_machines * gpus_per_node))
micro_batch=${MICRO_BATCH:-$default_micro_batch}
global_sequences=${GLOBAL_SEQUENCES:-2048}
sequence_length=1024
training_steps=620
warmup_steps=500

denominator=$((world_size * micro_batch))
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    global_sequences=$denominator
    training_steps=2
    warmup_steps=1
fi
if ((global_sequences % denominator != 0)); then
    echo "GLOBAL_SEQUENCES=$global_sequences must be divisible by WORLD_SIZE*MICRO_BATCH=$denominator" >&2
    exit 2
fi
gradient_accumulation_steps=$((global_sequences / denominator))

if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    else
        MASTER_ADDR=127.0.0.1
    fi
fi
MASTER_PORT=${MASTER_PORT:-29500}

dataset_path=${OPTIBERT_DATASET:-$project_root/tokenized_datasets/fineweb_edu_roberta_1p6b}
run_suffix=$variant
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    run_suffix="$variant-smoke"
fi
run_root=${RUN_ROOT:-$project_root/logs/optibertneo-1p3b/$run_suffix}
accelerate_config=${ACCELERATE_CONFIG:-$project_root/conf/accelerate_deepspeed_zero2.yaml}
wandb_mode=${WANDB_MODE:-online}
compile_model=${TORCH_COMPILE:-true}

echo "variant=$variant model=$model_config"
echo "world_size=$world_size micro_batch=$micro_batch gradient_accumulation=$gradient_accumulation_steps"
echo "global_batch=$global_sequences sequences ($((global_sequences * sequence_length)) tokens)"
echo "steps=$training_steps scheduled_tokens=$((training_steps * global_sequences * sequence_length))"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

exec accelerate launch \
    --config_file="$accelerate_config" \
    --machine_rank="$machine_rank" \
    --num_processes="$world_size" \
    --num_machines="$num_machines" \
    --main_process_ip="$MASTER_ADDR" \
    --main_process_port="$MASTER_PORT" \
    "$project_root/scripts/pretraining/pretrain.py" \
    dataset=fineweb_edu \
    tokenizer=roberta \
    model="$model_config" \
    datacollator=mlm_20 \
    optimizer=optibertneo \
    scheduler=optibertneo_1p3b \
    trainer=optibertneo_1p3b \
    dataloader=optibertneo \
    dataset.path_to_disk="$dataset_path" \
    dataloader.train.batch_size="$micro_batch" \
    trainer.gradient_accumulation_steps="$gradient_accumulation_steps" \
    trainer.max_steps="$training_steps" \
    scheduler.warmup_steps="$warmup_steps" \
    scheduler.decay_steps="$training_steps" \
    trainer.compile="$compile_model" \
    trainer.dir="$run_root" \
    wandb.name="optibertneo-1p3b-$variant" \
    wandb.project=complex-optibertneo \
    wandb.mode="$wandb_mode" \
    wandb.dir="$run_root/wandb" \
    hydra.run.dir="$run_root/hydra"
