#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: jobs/optibertneo-1p3b.sh {real|baseline|mixed}

Launch this script once per node. For the paper's real-valued OptiBERTneo
1.3B-token run, use "real" (the legacy alias "baseline" is identical).

Important overrides:
  PYTHON_BIN, NUM_MACHINES, GPUS_PER_NODE, MACHINE_RANK
  MASTER_ADDR, MASTER_PORT, OPTIBERT_DATASET, RUN_ROOT
  MICRO_BATCH, GLOBAL_SEQUENCES, DATALOADER_WORKERS
  ACCELERATE_CONFIG, WANDB_MODE, WANDB_ENTITY, TORCH_COMPILE
  EXPECTED_WORLD_SIZE, EXPECTED_NUM_MACHINES, STAGE_DATASET
  SMOKE_TEST=1 (two optimizer steps), DRY_RUN=1 (print checks only)
EOF
    exit 2
}

require_positive_integer() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got: $value" >&2
        exit 2
    fi
}

variant=${1:-}
case "$variant" in
    real|baseline)
        run_variant=real
        model_config=optibertneo-198m
        default_micro_batch=32
        default_accelerate_config=accelerate_ddp.yaml
        ;;
    mixed)
        run_variant=mixed
        model_config=optibertneo-mixed-198m
        default_micro_batch=4
        default_accelerate_config=accelerate_deepspeed_zero2.yaml
        ;;
    *)
        usage
        ;;
esac

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
num_machines=${NUM_MACHINES:-${SLURM_JOB_NUM_NODES:-1}}
gpus_per_node=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
machine_rank=${MACHINE_RANK:-${SLURM_NODEID:-0}}
micro_batch=${MICRO_BATCH:-$default_micro_batch}
global_sequences=${GLOBAL_SEQUENCES:-2048}
sequence_length=1024
training_steps=620
warmup_steps=500

for pair in \
    "NUM_MACHINES:$num_machines" \
    "GPUS_PER_NODE:$gpus_per_node" \
    "MICRO_BATCH:$micro_batch" \
    "GLOBAL_SEQUENCES:$global_sequences"; do
    require_positive_integer "${pair%%:*}" "${pair#*:}"
done
if [[ ! "$machine_rank" =~ ^[0-9]+$ ]] || ((machine_rank >= num_machines)); then
    echo "MACHINE_RANK must be in [0, $((num_machines - 1))], got: $machine_rank" >&2
    exit 2
fi

world_size=$((num_machines * gpus_per_node))
if [[ -n "${EXPECTED_WORLD_SIZE:-}" ]] && ((world_size != EXPECTED_WORLD_SIZE)); then
    echo "Expected world size $EXPECTED_WORLD_SIZE, got $world_size" >&2
    exit 2
fi
if [[ -n "${EXPECTED_NUM_MACHINES:-}" ]] && ((num_machines != EXPECTED_NUM_MACHINES)); then
    echo "Expected $EXPECTED_NUM_MACHINES machines, got $num_machines" >&2
    exit 2
fi

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
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
    else
        MASTER_ADDR=127.0.0.1
    fi
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
    if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
        MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
    else
        MASTER_PORT=29500
    fi
fi
require_positive_integer MASTER_PORT "$MASTER_PORT"
if ((MASTER_PORT > 65535)); then
    echo "MASTER_PORT must be at most 65535, got: $MASTER_PORT" >&2
    exit 2
fi

dataset_path=${OPTIBERT_DATASET:-$project_root/tokenized_datasets/fineweb_edu_roberta_1p6b}
if [[ "${STAGE_DATASET:-0}" == 1 && "${DRY_RUN:-0}" != 1 ]]; then
    if [[ -z "${SLURM_TMPDIR:-}" ]]; then
        echo "STAGE_DATASET=1 requires SLURM_TMPDIR" >&2
        exit 2
    fi
    staged_dataset="$SLURM_TMPDIR/optibertneo-fineweb-edu"
    if [[ ! -f "$staged_dataset/dataset_info.json" ]]; then
        staged_tmp=$(mktemp -d "$SLURM_TMPDIR/optibertneo-data.XXXXXX")
        echo "Staging dataset on node $(hostname): $dataset_path -> $staged_tmp"
        cp -a "$dataset_path/." "$staged_tmp/"
        mv "$staged_tmp" "$staged_dataset"
    fi
    dataset_path=$staged_dataset
fi

run_suffix=$run_variant
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    run_suffix="$run_variant-smoke"
fi
run_root=${RUN_ROOT:-$project_root/logs/optibertneo-1p3b/$run_suffix}
accelerate_config=${ACCELERATE_CONFIG:-$project_root/conf/$default_accelerate_config}
wandb_mode=${WANDB_MODE:-offline}
compile_model=${TORCH_COMPILE:-true}
dataloader_workers=${DATALOADER_WORKERS:-8}
require_positive_integer DATALOADER_WORKERS "$dataloader_workers"

echo "variant=$run_variant model=$model_config"
echo "topology=${num_machines}x${gpus_per_node} world_size=$world_size machine_rank=$machine_rank"
echo "micro_batch=$micro_batch gradient_accumulation=$gradient_accumulation_steps"
echo "global_batch=$global_sequences sequences ($((global_sequences * sequence_length)) token positions)"
echo "steps=$training_steps scheduled_tokens=$((training_steps * global_sequences * sequence_length))"
echo "dataset=$dataset_path"
echo "run_root=$run_root"
echo "launcher=$accelerate_config python=$python_bin"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

if [[ ! -x "$python_bin" ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "PYTHON_BIN is not executable: $python_bin" >&2
    exit 1
fi
if [[ ! -f "$accelerate_config" ]]; then
    echo "Accelerate config does not exist: $accelerate_config" >&2
    exit 1
fi
if [[ ! -f "$dataset_path/dataset_info.json" ]]; then
    echo "Prepared dataset not found at $dataset_path" >&2
    echo "Run jobs/slurm/prepare-optibertneo-data.sbatch first." >&2
    exit 1
fi
mkdir -p "$run_root"

tokenizer_override=()
if [[ -f "$dataset_path/tokenizer/tokenizer.json" ]]; then
    tokenizer_override=(
        "tokenizer.pretrained_model_name_or_path=$dataset_path/tokenizer"
        "tokenizer.revision=null"
    )
fi

wandb_entity_override=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    wandb_entity_override=("wandb.entity=$WANDB_ENTITY")
fi

exec "$python_bin" -m accelerate.commands.launch \
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
    dataloader.train.num_workers="$dataloader_workers" \
    trainer.gradient_accumulation_steps="$gradient_accumulation_steps" \
    trainer.max_steps="$training_steps" \
    scheduler.warmup_steps="$warmup_steps" \
    scheduler.decay_steps="$training_steps" \
    trainer.compile="$compile_model" \
    trainer.dir="$run_root" \
    wandb.name="optibertneo-1p3b-$run_variant" \
    wandb.project=optibertneo \
    wandb.mode="$wandb_mode" \
    wandb.resume=allow \
    wandb.dir="$run_root/wandb" \
    hydra.run.dir="$run_root/hydra/\${oc.env:RANK,0}" \
    "${tokenizer_override[@]}" \
    "${wandb_entity_override[@]}"
