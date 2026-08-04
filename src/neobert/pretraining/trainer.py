import json
import os
import shutil
import re
from pathlib import Path
from tqdm import tqdm

from omegaconf import OmegaConf, DictConfig

# PyTorch
import torch
from torch.nn import CrossEntropyLoss

# Hugging Face
from datasets import load_from_disk
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


def export_pretrained_model(accelerator, model, tokenizer, cfg, metrics):
    export_path = Path(cfg.trainer.dir) / "final_model"
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    state_dict = _state_dict_without_compile_prefix(state_dict)
    if accelerator.is_main_process:
        model_to_save = _model_without_compile_wrapper(accelerator, model)
        model_to_save.save_pretrained(
            export_path,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(export_path)
        (export_path / "training_summary.json").write_text(
            json.dumps(
                {
                    "optimizer_steps": int(metrics["train/steps"]),
                    "training_sequences": int(metrics["train/samples"]),
                    "training_tokens": int(metrics["train/tokens"]),
                    "masked_tokens": int(metrics["train/masked_tokens"]),
                    "resolved_config": OmegaConf.to_container(cfg, resolve=True),
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


def trainer(cfg: DictConfig):
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
    os.makedirs(cfg.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": OmegaConf.to_container(cfg) | {"distributed_type": accelerator.distributed_type},
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "resume": cfg.wandb.resume,
            }
        },
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
    train_dataset = load_from_disk(cfg.dataset.path_to_disk)
    if cfg.dataloader.train.get("prepacked_sequences", False):
        validate_prepacked_dataset(
            train_dataset,
            tokenizer,
            cfg,
            accelerator.num_processes,
        )

    # Dataloader
    train_dataloader = get_dataloader(train_dataset, tokenizer, dtype=dtype_pad_mask, **cfg.dataloader.train, **cfg.datacollator)

    # Model
    model = NeoBERTLMHead(NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id))
    embedding = model.model.encoder.weight
    accelerator.log(
        {
            "model/parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model/non_embedding_parameters": sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad and p is not embedding
            ),
        }
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
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader,
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

                if metrics["train/steps"] % cfg.wandb.log_interval == 0:
                    # https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.utils.safe_get_full_grad
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        metrics["train/grad_norm"] = model.get_global_grad_norm()
                        metrics["train/weight_norm"] = (
                            sum([safe_get_full_fp32_param(p).norm(2) ** 2 for p in model.parameters()]) ** 0.5
                        ).item()
                    # DDP
                    else:
                        metrics["train/grad_norm"] = (sum([p.grad.norm(2) ** 2 for p in model.parameters()]) ** 0.5).item()
                        metrics["train/weight_norm"] = (sum([p.norm(2) ** 2 for p in model.parameters()]) ** 0.5).item()

                    metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                    metrics.log(accelerator)

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

                if metrics["train/steps"] >= cfg.trainer.max_steps:
                    break

                # Reset the gradient
                optimizer.zero_grad()

        # Log metrics
        metrics["train/epochs"] += 1

        # "Remove" the skipped dataloader once exhausted
        skipped_train_dataloader = None

    if metrics["train/local_num_pred"] > 0:
        metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
        metrics.log(accelerator)

    if (
        cfg.trainer.model.get("save_at_end", False)
        and metrics["train/steps"] % cfg.trainer.model.save_steps != 0
    ):
        save_accelerator_checkpoint(accelerator)
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
