import json
import math
import os
import shutil
import re
import time
from pathlib import Path
from tqdm import tqdm

from omegaconf import OmegaConf, DictConfig

# PyTorch
import torch
from torch.nn import CrossEntropyLoss

# Hugging Face
from datasets import DatasetDict, load_from_disk
from transformers import BatchEncoding
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param

# Our metric object and model
from .metrics import Metrics
from ..model import NeoBERTLMHead, NeoBERTConfig
from ..tokenizer import get_tokenizer
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..dataloader import get_dataloader


def to_target_batch_size(
    batch: BatchEncoding,
    stored_batch: BatchEncoding,
    target_size: int = 8,
):
    if stored_batch:
        if batch.keys() != stored_batch.keys():
            raise ValueError("stored and current batches must contain the same fields")
        batch = {
            key: torch.cat((stored_batch[key].to(value.device), value), dim=0)
            for key, value in batch.items()
        }

    batch_size = batch["input_ids"].shape[0]
    if batch_size <= target_size:
        return batch, {}

    output = {}
    remainder = {}
    for key, value in batch.items():
        output[key], remainder[key] = value.split(
            (target_size, batch_size - target_size),
            dim=0,
        )
        remainder[key] = remainder[key].cpu()
    return output, remainder


def count_batch_tokens(batch: BatchEncoding) -> int:
    if "document_ids" in batch:
        return (batch["document_ids"] >= 0).sum().item()
    if "attention_mask" in batch:
        return (batch["attention_mask"] == 0).sum().item()
    return batch["input_ids"].numel()


def split_train_validation_dataset(dataset):
    """Return train/validation datasets while preserving legacy Dataset runs."""
    if not isinstance(dataset, DatasetDict):
        return dataset, None
    if "train" not in dataset:
        raise ValueError("a saved DatasetDict must contain a 'train' split")
    return dataset["train"], dataset.get("validation")


def validate_flat_packed_dataset_dict(dataset, cfg):
    """Validate the mask-free fixed-row DatasetDict used by the ablation."""
    if not cfg.dataset.get("cross_document_attention", False):
        return

    configured_length = cfg.dataset.get("pack_to_length")
    if configured_length is None:
        raise ValueError(
            "dataset.cross_document_attention=true requires dataset.pack_to_length"
        )
    sequence_length = int(configured_length)
    if sequence_length <= 0:
        raise ValueError("dataset.pack_to_length must be positive")
    if not isinstance(dataset, DatasetDict):
        raise ValueError(
            "cross-document ablation data must be a DatasetDict with train and validation splits"
        )
    expected_splits = {"train", "validation"}
    if set(dataset) != expected_splits:
        raise ValueError(
            "cross-document ablation data must contain exactly train and validation splits"
        )

    for split_name in sorted(expected_splits):
        split_dataset = dataset[split_name]
        if split_dataset.column_names != ["input_ids"]:
            raise ValueError(
                f"{split_name} must contain only input_ids; found "
                f"{split_dataset.column_names}"
            )
        if len(split_dataset) == 0:
            raise ValueError(f"{split_name} must contain at least one packed row")
        sampled_indices = {0, len(split_dataset) // 2, len(split_dataset) - 1}
        for index in sampled_indices:
            row_length = len(split_dataset[index]["input_ids"])
            if row_length != sequence_length:
                raise ValueError(
                    f"{split_name} row {index} has {row_length} tokens; "
                    f"expected {sequence_length}"
                )

    manifest_path = Path(cfg.dataset.path_to_disk) / "optibertneo_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sequence_length") != sequence_length:
        raise ValueError("dataset manifest sequence length does not match the run")
    packing = manifest.get("packing", {})
    if packing.get("padding_free") is not True:
        raise ValueError("dataset manifest must declare padding_free=true")
    if packing.get("cross_document_attention") is not True:
        raise ValueError(
            "dataset manifest must declare cross_document_attention=true"
        )
    if packing.get("document_ids") is not False:
        raise ValueError("dataset manifest must declare document_ids=false")

    manifest_splits = manifest.get("splits", {})
    if set(manifest_splits) != expected_splits:
        raise ValueError(
            "dataset manifest must describe exactly train and validation splits"
        )
    for split_name in sorted(expected_splits):
        split_dataset = dataset[split_name]
        split_manifest = manifest_splits[split_name]
        expected_tokens = len(split_dataset) * sequence_length
        if split_manifest.get("rows") != len(split_dataset):
            raise ValueError(
                f"dataset manifest {split_name} row count does not match Arrow data"
            )
        if split_manifest.get("tokens") != expected_tokens:
            raise ValueError(
                f"dataset manifest {split_name} token count does not match Arrow data"
            )
        if split_manifest.get("packed_token_positions") != expected_tokens:
            raise ValueError(
                f"dataset manifest {split_name} packed token count does not match Arrow data"
            )
        if split_manifest.get("columns") != ["input_ids"]:
            raise ValueError(
                f"dataset manifest {split_name} columns must contain only input_ids"
            )


def validation_dataloader_kwargs(cfg):
    """Derive a deterministic, main-process-collated validation loader."""
    options = OmegaConf.to_container(cfg.dataloader.train, resolve=True)
    configured = cfg.dataloader.get("validation")
    if configured is not None:
        options.update(OmegaConf.to_container(configured, resolve=True))
    options.update(
        {
            "shuffle": False,
            "num_workers": 0,
            "persistent_workers": False,
        }
    )
    return options


def _sanitise_wandb_identifier(value):
    if value is None:
        return None
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")
    return value[:128] or None


def runtime_metadata(accelerator):
    """Return a small, credential-free runtime description for W&B config."""
    metadata = {
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "distributed_type": str(accelerator.distributed_type),
        "num_processes": int(accelerator.num_processes),
        "mixed_precision": str(accelerator.mixed_precision),
    }
    slurm_names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_CLUSTER_NAME",
        "SLURM_NNODES",
        "SLURM_GPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
    )
    metadata["slurm"] = {
        name.removeprefix("SLURM_").lower(): os.environ[name]
        for name in slurm_names
        if name in os.environ
    }
    if torch.cuda.is_available():
        device_index = accelerator.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        metadata["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return metadata


def wandb_init_kwargs(cfg, accelerator):
    """Build reproducible W&B identity and serializable run configuration."""
    name = cfg.wandb.get("name")
    explicit_id = cfg.wandb.get("id") or os.environ.get("WANDB_RUN_ID")
    slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get(
        "SLURM_JOB_ID"
    )
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if explicit_id is None and slurm_job_id is not None:
        identity_parts = [slurm_job_id]
        if array_task_id is not None:
            identity_parts.append(array_task_id)
        identity_parts.append(name or os.environ.get("SLURM_JOB_NAME") or "train")
        explicit_id = "-".join(identity_parts)

    group = cfg.wandb.get("group") or os.environ.get("WANDB_RUN_GROUP")
    if group is None:
        group = slurm_job_id
    job_type = (
        cfg.wandb.get("job_type")
        or os.environ.get("WANDB_JOB_TYPE")
        or "training"
    )

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    resolved_config["runtime"] = runtime_metadata(accelerator)
    options = {
        "name": name,
        "entity": cfg.wandb.get("entity"),
        "config": resolved_config,
        "tags": list(cfg.wandb.get("tags", [])),
        "dir": cfg.wandb.get("dir"),
        "mode": cfg.wandb.get("mode", "online"),
        "resume": cfg.wandb.get("resume", "allow"),
        "id": _sanitise_wandb_identifier(explicit_id),
        "group": _sanitise_wandb_identifier(group),
        "job_type": _sanitise_wandb_identifier(job_type),
    }
    return {key: value for key, value in options.items() if value is not None}


def _synchronise_cuda(accelerator):
    if accelerator.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)


def performance_metrics(
    accelerator,
    metrics,
    *,
    interval_started_at,
    previous_step,
    run_started_at,
):
    """Measure rank-global training throughput over the current log window."""
    _synchronise_cuda(accelerator)
    elapsed = max(time.monotonic() - interval_started_at, 1e-12)
    local_counts = torch.tensor(
        [
            metrics.get("train/local_tokens", 0),
            metrics.get("train/local_samples", 0),
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )
    global_counts = accelerator.reduce(local_counts, reduction="sum")
    elapsed_tensor = torch.tensor(
        elapsed,
        dtype=torch.float64,
        device=accelerator.device,
    )
    elapsed = float(
        accelerator.reduce(elapsed_tensor, reduction="mean").detach().cpu()
    )
    global_tokens, global_samples = (
        global_counts.detach().cpu().tolist()
    )
    interval_steps = max(int(metrics["train/steps"]) - previous_step, 1)
    output = {
        "performance/tokens_per_second": global_tokens / elapsed,
        "performance/sequences_per_second": global_samples / elapsed,
        "performance/step_time_ms": 1000.0 * elapsed / interval_steps,
        "performance/window_seconds": elapsed,
        "performance/wall_time_seconds": time.monotonic() - run_started_at,
    }
    if accelerator.device.type == "cuda" and torch.cuda.is_available():
        output.update(
            {
                "system/peak_cuda_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(accelerator.device)
                ),
                "system/peak_cuda_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(accelerator.device)
                ),
            }
        )
    return output


def max_time_reached(accelerator, run_started_at, max_time_seconds):
    if max_time_seconds is None:
        return False
    local_reached = float(time.monotonic() - run_started_at >= max_time_seconds)
    reached = torch.tensor(
        local_reached,
        dtype=torch.float32,
        device=accelerator.device,
    )
    # A sum implements an any-rank stop decision with Accelerate's portable
    # reduction API (which supports sum/mean, but not a cross-backend max).
    reached = accelerator.reduce(reached, reduction="sum")
    return bool(reached.detach().cpu().item())


def update_model_norm_metrics(metrics, accelerator, model):
    """Update scalar gradient/weight norms immediately before tracker logging."""
    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        grad_norm = model.get_global_grad_norm()
        metrics["train/grad_norm"] = (
            grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
        )
        full_parameters = [
            parameter
            for parameter in (
                safe_get_full_fp32_param(parameter)
                for parameter in model.parameters()
            )
            if parameter is not None
        ]
        metrics["train/weight_norm"] = (
            sum(parameter.norm(2) ** 2 for parameter in full_parameters) ** 0.5
        ).item()
        return

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    metrics["train/grad_norm"] = (
        sum(gradient.norm(2) ** 2 for gradient in gradients) ** 0.5
    ).item() if gradients else 0.0
    metrics["train/weight_norm"] = (
        sum(parameter.norm(2) ** 2 for parameter in model.parameters()) ** 0.5
    ).item()


def evaluate_mlm(
    accelerator,
    model,
    dataloader,
    *,
    seed,
    max_batches=None,
):
    """Evaluate one deterministic set of held-out MLM corruptions."""
    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            raise ValueError("trainer.validation.max_batches must be positive")

    was_training = model.training
    totals = torch.zeros(5, dtype=torch.float64, device=accelerator.device)
    _synchronise_cuda(accelerator)
    started_at = time.monotonic()
    try:
        model.eval()
        # Validation collation runs with num_workers=0, so restoring the CPU RNG
        # makes both masking and the caller's training RNG state deterministic.
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(int(seed) + int(accelerator.process_index))
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                logits = model(
                    batch["input_ids"],
                    batch.get("attention_mask"),
                    batch.get("document_ids"),
                )["logits"]
                labels = batch["labels"]
                selected = labels != -100
                loss_sum = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                totals[0] += loss_sum.detach().to(torch.float64)
                totals[1] += ((logits.argmax(dim=-1) == labels) & selected).sum()
                totals[2] += selected.sum()
                totals[3] += batch["input_ids"].shape[0]
                totals[4] += count_batch_tokens(batch)
    finally:
        _synchronise_cuda(accelerator)
        model.train(was_training)

    totals = accelerator.reduce(totals, reduction="sum").detach().cpu().tolist()
    loss_sum, num_correct, num_pred, samples, tokens = totals
    if num_pred > 0:
        loss = loss_sum / num_pred
        perplexity = math.exp(loss) if loss < 709.0 else float("inf")
        accuracy = num_correct / num_pred
    else:
        loss = float("nan")
        perplexity = float("nan")
        accuracy = float("nan")
    return {
        "validation/mlm_loss": loss,
        "validation/mlm_perplexity": perplexity,
        "validation/masked_accuracy": accuracy,
        "validation/samples": int(samples),
        "validation/tokens": int(tokens),
        "validation/masked_tokens": int(num_pred),
        "validation/wall_time_seconds": time.monotonic() - started_at,
    }


def validate_prepacked_dataset(dataset, tokenizer, cfg, world_size):
    required_columns = {"input_ids", "document_ids"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(
            "prepacked dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    sequence_length = int(cfg.tokenizer.max_length)
    required_rows = (
        int(cfg.trainer.max_steps)
        * int(cfg.dataloader.train.batch_size)
        * int(cfg.trainer.gradient_accumulation_steps)
        * int(world_size)
    )
    if len(dataset) < required_rows:
        raise ValueError(
            f"prepared dataset has {len(dataset):,} rows, but this run needs "
            f"at least {required_rows:,} rows to avoid repeating data"
        )

    for index in {0, len(dataset) - 1}:
        row = dataset[index]
        if len(row["input_ids"]) != sequence_length:
            raise ValueError(
                f"dataset row {index} has {len(row['input_ids'])} tokens; "
                f"expected {sequence_length}"
            )
        if len(row["document_ids"]) != sequence_length:
            raise ValueError(
                f"dataset row {index} has {len(row['document_ids'])} document ids; "
                f"expected {sequence_length}"
            )
        if any(document_id < 0 for document_id in row["document_ids"]):
            raise ValueError(
                f"dataset row {index} contains padding document ids; "
                "the OptiBERTneo paper recipe is padding-free"
            )

    manifest_path = Path(cfg.dataset.path_to_disk) / "optibertneo_manifest.json"
    if not manifest_path.is_file():
        if cfg.dataset.get("require_manifest", False):
            raise ValueError(
                f"dataset manifest is missing: {manifest_path}. "
                "Re-run scripts/pretraining/preprocess.py with the OptiBERTneo setup."
            )
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_positions = len(dataset) * sequence_length
    if manifest.get("rows") != len(dataset):
        raise ValueError("dataset manifest row count does not match the Arrow dataset")
    if manifest.get("sequence_length") != sequence_length:
        raise ValueError("dataset manifest sequence length does not match the run")
    if manifest.get("packed_token_positions") != expected_positions:
        raise ValueError("dataset manifest token count does not match the Arrow dataset")
    if not manifest.get("packing", {}).get("padding_free", False):
        raise ValueError("dataset manifest does not declare padding-free packing")
    tokenizer_manifest = manifest.get("tokenizer", {})
    expected_tokenizer_values = {
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
    }
    for name, expected in expected_tokenizer_values.items():
        if tokenizer_manifest.get(name) != expected:
            raise ValueError(
                f"dataset tokenizer {name}={tokenizer_manifest.get(name)!r} "
                f"does not match the training tokenizer value {expected!r}"
            )


def _state_dict_without_compile_prefix(state_dict):
    prefix = "_orig_mod."
    while state_dict and all(key.startswith(prefix) for key in state_dict):
        state_dict = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
        }
    return state_dict


def _model_without_compile_wrapper(accelerator, model):
    model = accelerator.unwrap_model(model)
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def discover_resume_checkpoint(checkpoint_directory):
    """Return the latest complete Accelerate checkpoint and next save index."""
    checkpoint_directory = Path(checkpoint_directory)
    indexed_checkpoints = []
    if checkpoint_directory.is_dir():
        for path in checkpoint_directory.iterdir():
            match = re.fullmatch(r"checkpoint_(\d+)", path.name)
            if path.is_dir() and match is not None:
                indexed_checkpoints.append((int(match.group(1)), path))

    next_iteration = (
        max(index for index, _ in indexed_checkpoints) + 1
        if indexed_checkpoints
        else 0
    )
    complete_checkpoints = [
        (index, path)
        for index, path in indexed_checkpoints
        if (path / "_SUCCESS").is_file()
    ]
    latest_complete = (
        max(complete_checkpoints, key=lambda item: item[0])[1]
        if complete_checkpoints
        else None
    )
    return latest_complete, next_iteration


def save_accelerator_checkpoint(accelerator):
    """Save a resumable state and mark it complete only after every rank exits."""
    checkpoint_path = (
        Path(accelerator.project_dir)
        / "checkpoints"
        / f"checkpoint_{accelerator.save_iteration}"
    )
    accelerator.save_state()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (checkpoint_path / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    accelerator.wait_for_everyone()


def save_model_checkpoint(accelerator, model, directory, step):
    checkpoint_path = Path(directory) / str(step)
    accelerator.wait_for_everyone()
    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        model.save_checkpoint(str(directory), tag=str(step))
    elif accelerator.is_main_process:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        state_dict = _state_dict_without_compile_prefix(
            accelerator.get_state_dict(model)
        )
        temporary_path = checkpoint_path / "state_dict.pt.tmp"
        torch.save(state_dict, temporary_path)
        os.replace(temporary_path, checkpoint_path / "state_dict.pt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (checkpoint_path / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    accelerator.wait_for_everyone()


def ensure_babylm_auto_map(config):
    """Advertise all Auto classes consumed by the official BabyLM harness."""
    auto_map = dict(getattr(config, "auto_map", None) or {})
    auto_map.update(
        {
            "AutoConfig": "model.NeoBERTConfig",
            "AutoModelForMaskedLM": "model.NeoBERTLMHead",
            "AutoModelForSequenceClassification": (
                "model.NeoBERTHFForSequenceClassification"
            ),
        }
    )
    config.auto_map = auto_map


def build_training_summary(cfg, metrics):
    return {
        "optimizer_steps": int(metrics["train/steps"]),
        "training_sequences": int(metrics["train/samples"]),
        "training_tokens": int(metrics["train/tokens"]),
        "masked_tokens": int(metrics["train/masked_tokens"]),
        "train/completed_schedule": (
            int(metrics["train/steps"]) == int(cfg.trainer.max_steps)
        ),
        "train/stopped_for_max_time": bool(
            metrics["train/stopped_for_max_time"]
        ),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }


def export_pretrained_model(accelerator, model, tokenizer, cfg, metrics):
    export_path = Path(cfg.trainer.dir) / "final_model"
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    state_dict = _state_dict_without_compile_prefix(state_dict)
    if accelerator.is_main_process:
        model_to_save = _model_without_compile_wrapper(accelerator, model)
        ensure_babylm_auto_map(model_to_save.config)
        model_to_save.save_pretrained(
            export_path,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(export_path)
        (export_path / "training_summary.json").write_text(
            json.dumps(
                build_training_summary(cfg, metrics),
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


def trainer(cfg: DictConfig):
    run_started_at = time.monotonic()
    log_interval = int(cfg.wandb.log_interval)
    if log_interval <= 0:
        raise ValueError("wandb.log_interval must be positive")
    max_time_seconds = cfg.trainer.get("max_time_seconds")
    if max_time_seconds is not None:
        max_time_seconds = float(max_time_seconds)
        if not math.isfinite(max_time_seconds) or max_time_seconds <= 0:
            raise ValueError("trainer.max_time_seconds must be a positive finite number")

    # Get the last complete checkpoint and choose an unused save index. An
    # interrupted checkpoint remains on disk for diagnosis but is never loaded.
    checkpoint_dir = os.path.join(cfg.trainer.dir, "checkpoints")
    model_checkpoint_dir = os.path.join(cfg.trainer.dir, "model_checkpoints")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    resume_checkpoint, iteration = discover_resume_checkpoint(checkpoint_dir)
    if (
        cfg.trainer.resume
        and os.path.isdir(checkpoint_dir)
        and any(Path(checkpoint_dir).iterdir())
        and resume_checkpoint is None
    ):
        raise RuntimeError(
            f"{checkpoint_dir} contains no completed checkpoint. "
            "Keep it for diagnosis and select a fresh RUN_ROOT."
        )

    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.trainer.accelerate.max_ckpt,
        iteration=iteration,
    )
    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.trainer.get(
            "find_unused_parameters",
            True,
        )
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb",
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    # Initialise the wandb run and pass wandb parameters
    if cfg.wandb.get("dir") is not None:
        os.makedirs(cfg.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={"wandb": wandb_init_kwargs(cfg, accelerator)},
    )

    # Set the seed
    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg.trainer.tf32

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    # Tokenizer
    tokenizer = get_tokenizer(**cfg.tokenizer)

    # Dataset
    loaded_dataset = load_from_disk(cfg.dataset.path_to_disk)
    validate_flat_packed_dataset_dict(loaded_dataset, cfg)
    train_dataset, validation_dataset = split_train_validation_dataset(
        loaded_dataset
    )
    if cfg.dataloader.train.get("prepacked_sequences", False):
        validate_prepacked_dataset(
            train_dataset,
            tokenizer,
            cfg,
            accelerator.num_processes,
        )

    # Dataloaders. Validation deliberately collates in the training process so
    # evaluate_mlm can restore an identical MLM RNG stream on every invocation.
    validation_cfg = cfg.trainer.get("validation", {})
    validation_eval_steps = int(
        validation_cfg.get("eval_steps", log_interval)
    )
    if validation_eval_steps <= 0:
        raise ValueError("trainer.validation.eval_steps must be positive")
    validation_max_batches = validation_cfg.get("max_batches")
    validation_seed = int(validation_cfg.get("seed", cfg.seed + 10_000))
    train_loader_options = OmegaConf.to_container(
        cfg.dataloader.train,
        resolve=True,
    )
    train_loader_options.setdefault("seed", int(cfg.seed))
    train_dataloader = get_dataloader(
        train_dataset,
        tokenizer,
        dtype=dtype_pad_mask,
        **train_loader_options,
        **cfg.datacollator,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_loader_options = validation_dataloader_kwargs(cfg)
        validation_loader_options.setdefault("seed", validation_seed)
        validation_dataloader = get_dataloader(
            validation_dataset,
            tokenizer,
            dtype=dtype_pad_mask,
            **validation_loader_options,
            **cfg.datacollator,
        )

    # Model
    model = NeoBERTLMHead(NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id))
    embedding = model.model.encoder.weight
    if resume_checkpoint is None:
        accelerator.log(
            {
                "model/parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "model/non_embedding_parameters": sum(
                    p.numel()
                    for p in model.parameters()
                    if p.requires_grad and p is not embedding
                ),
            },
            step=0,
        )
    if cfg.trainer.get("compile", False):
        model = torch.compile(
            model,
            fullgraph=cfg.trainer.get("compile_fullgraph", False),
        )

    # Optimizer and Scheduler
    optimizer = get_optimizer(model, accelerator.distributed_type, name=cfg.optimizer.name, **cfg.optimizer.hparams)
    scheduler = get_scheduler(optimizer=optimizer, lr=cfg.optimizer.hparams.lr, **cfg.scheduler)

    # Prepare with accelerate
    if validation_dataloader is None:
        train_dataloader, model, optimizer, scheduler = accelerator.prepare(
            train_dataloader,
            model,
            optimizer,
            scheduler,
        )
    else:
        (
            train_dataloader,
            validation_dataloader,
            model,
            optimizer,
            scheduler,
        ) = accelerator.prepare(
            train_dataloader,
            validation_dataloader,
            model,
            optimizer,
            scheduler,
        )

    # Loss function
    train_loss_fn = CrossEntropyLoss()

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume and resume_checkpoint is not None:
        accelerator.load_state(str(resume_checkpoint))
        train_dataloader.set_epoch(metrics["train/epochs"])
        skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["train/batches"] % len(train_dataloader))

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["train/steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    interval_started_at = time.monotonic()
    previous_log_step = int(metrics["train/steps"])
    last_eval_step = None
    stopped_for_max_time = False
    final_status_logged = False
    # A resumed run starts a new wall-clock guard window. Historical partial
    # stops remain in W&B history, while the final export describes this run.
    metrics["train/completed_schedule"] = int(
        metrics["train/steps"] == cfg.trainer.max_steps
    )
    metrics["train/stopped_for_max_time"] = 0
    if accelerator.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    while cfg.trainer.max_steps > metrics["train/steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = train_dataloader if skipped_train_dataloader is None else skipped_train_dataloader

        stored_batch = {}
        i = 0
        for batch in dataloader:
            # Update number of batches
            metrics["train/batches"] += 1
            i += 1

            # Pack or truncate the batch to target batch size (batch size might be variable due to sequence packing).
            if batch["input_ids"].shape[0] != cfg.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg.dataloader.train.batch_size)

            # If it is still smaller, stored batches were not enough and we skip to the next iteration to fill the batch
            if batch["input_ids"].shape[0] < cfg.dataloader.train.batch_size:
                stored_batch = batch
                continue

            # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
            # called, and the first call to .backward() outside this context manager will trigger the synchronization.
            # Accumulating manually gives more flexibility and is compatible with TPUs.
            if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):
                    # Forward pass
                    logits = model(
                        batch["input_ids"],
                        batch.get("attention_mask"),
                        batch.get("document_ids"),
                    )["logits"]
                    train_loss = train_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))

                    # Compute gradient
                    accelerator.backward(train_loss)

                    # Log metrics
                    metrics["train/local_samples"] += batch["input_ids"].shape[0]
                    metrics["train/local_tokens"] += count_batch_tokens(batch)
                    metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
                    metrics["train/local_sum_loss"] += train_loss.item() * (batch["labels"] != -100).sum().item()
                    metrics["train/local_num_correct"] += (logits.argmax(dim=-1) == batch["labels"]).sum().item()

            else:
                # Forward pass
                logits = model(
                    batch["input_ids"],
                    batch.get("attention_mask"),
                    batch.get("document_ids"),
                )["logits"]
                train_loss = train_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))

                # Compute gradient and apply clipping
                accelerator.backward(train_loss)
                if cfg.trainer.gradient_clipping is not None and cfg.trainer.gradient_clipping > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)

                # Log metrics
                pbar.update(1)
                metrics["train/steps"] += 1
                metrics["train/local_samples"] += batch["input_ids"].shape[0]
                metrics["train/local_tokens"] += count_batch_tokens(batch)
                metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
                metrics["train/local_sum_loss"] += train_loss.item() * (batch["labels"] != -100).sum().item()
                metrics["train/local_num_correct"] += (logits.argmax(dim=-1) == batch["labels"]).sum().item()

                # Update the parameters and the scheduler
                optimizer.step()
                scheduler.step()

                reached_step_limit = (
                    metrics["train/steps"] >= cfg.trainer.max_steps
                )
                reached_time_limit = max_time_reached(
                    accelerator,
                    run_started_at,
                    max_time_seconds,
                )
                stopped_for_max_time = (
                    stopped_for_max_time or reached_time_limit
                )
                should_stop = reached_step_limit or reached_time_limit
                if should_stop:
                    metrics["train/completed_schedule"] = int(
                        metrics["train/steps"] == cfg.trainer.max_steps
                    )
                    metrics["train/stopped_for_max_time"] = int(
                        reached_time_limit
                    )
                should_evaluate = validation_dataloader is not None and (
                    metrics["train/steps"] % validation_eval_steps == 0
                    or should_stop
                )
                should_log = (
                    metrics["train/steps"] % log_interval == 0
                    or should_evaluate
                    or should_stop
                )

                if should_log:
                    update_model_norm_metrics(metrics, accelerator, model)
                    metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                    extra_metrics = performance_metrics(
                        accelerator,
                        metrics,
                        interval_started_at=interval_started_at,
                        previous_step=previous_log_step,
                        run_started_at=run_started_at,
                    )
                    if should_evaluate:
                        extra_metrics.update(
                            evaluate_mlm(
                                accelerator,
                                model,
                                validation_dataloader,
                                seed=validation_seed,
                                max_batches=validation_max_batches,
                            )
                        )
                        last_eval_step = int(metrics["train/steps"])
                    if reached_time_limit:
                        extra_metrics["train/stopped_for_max_time"] = 1
                    metrics.log(
                        accelerator,
                        step=int(metrics["train/steps"]),
                        extra_metrics=extra_metrics,
                    )
                    if should_stop:
                        final_status_logged = True
                    _synchronise_cuda(accelerator)
                    if (
                        accelerator.device.type == "cuda"
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.reset_peak_memory_stats(accelerator.device)
                    interval_started_at = time.monotonic()
                    previous_log_step = int(metrics["train/steps"])

                # Save the accelerator state from the main process
                if metrics["train/steps"] % cfg.trainer.accelerate.save_steps == 0:
                    save_accelerator_checkpoint(accelerator)

                # Save the pytorch model
                if metrics["train/steps"] % cfg.trainer.model.save_steps == 0:
                    if (
                        cfg.trainer.model.max_ckpt is not None
                        and accelerator.is_main_process
                    ):
                        # Delete checkpoints if there are too many
                        files = os.listdir(model_checkpoint_dir)
                        iterations = [int(f) for f in files if f.isdigit()]
                        iterations.sort()

                        # Remove files with the smallest iterations until the limit is met
                        while iterations is not None and len(iterations) >= cfg.trainer.model.max_ckpt:
                            file_to_remove = iterations.pop(0)
                            shutil.rmtree(os.path.join(model_checkpoint_dir, str(file_to_remove)))
                            print(
                                 f"Deleted old model checkpoint {file_to_remove} due to limit " f"(max_ckpt = {cfg.trainer.model.max_ckpt})"
                            )
                    accelerator.wait_for_everyone()
                    save_model_checkpoint(
                        accelerator,
                        model,
                        model_checkpoint_dir,
                        metrics["train/steps"],
                    )

                if should_stop:
                    break

                # Reset the gradient
                optimizer.zero_grad()

        # Log metrics
        metrics["train/epochs"] += 1

        # "Remove" the skipped dataloader once exhausted
        skipped_train_dataloader = None

        if stopped_for_max_time:
            break

    final_metrics = {}
    metrics["train/completed_schedule"] = int(
        metrics["train/steps"] == cfg.trainer.max_steps
    )
    metrics["train/stopped_for_max_time"] = int(stopped_for_max_time)
    if not final_status_logged:
        # These duplicate the persistent metric keys intentionally: adding them
        # here ensures a status-only final record is emitted after a no-op resume.
        final_metrics["train/completed_schedule"] = metrics[
            "train/completed_schedule"
        ]
        final_metrics["train/stopped_for_max_time"] = metrics[
            "train/stopped_for_max_time"
        ]
    if (
        validation_dataloader is not None
        and last_eval_step != int(metrics["train/steps"])
    ):
        final_metrics.update(
            evaluate_mlm(
                accelerator,
                model,
                validation_dataloader,
                seed=validation_seed,
                max_batches=validation_max_batches,
            )
        )
        last_eval_step = int(metrics["train/steps"])
    if metrics.get("train/local_num_pred", 0) > 0:
        update_model_norm_metrics(metrics, accelerator, model)
        metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
        final_metrics.update(
            performance_metrics(
                accelerator,
                metrics,
                interval_started_at=interval_started_at,
                previous_step=previous_log_step,
                run_started_at=run_started_at,
            )
        )
    if metrics.get("train/local_num_pred", 0) > 0 or final_metrics:
        metrics.log(
            accelerator,
            step=int(metrics["train/steps"]),
            extra_metrics=final_metrics,
        )

    if cfg.trainer.model.get("save_at_end", False) or stopped_for_max_time:
        if metrics["train/steps"] % cfg.trainer.accelerate.save_steps != 0:
            save_accelerator_checkpoint(accelerator)
        if metrics["train/steps"] % cfg.trainer.model.save_steps != 0:
            save_model_checkpoint(
                accelerator,
                model,
                model_checkpoint_dir,
                metrics["train/steps"],
            )

    if cfg.trainer.model.get("export_at_end", False):
        export_pretrained_model(
            accelerator,
            model,
            tokenizer,
            cfg,
            metrics,
        )

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()
