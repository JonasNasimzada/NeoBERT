#!/usr/bin/env python3
"""Paired external LRA-CIFAR10 transfer evaluation for FineWeb checkpoints.

The official LRA Image task presents 1,024 grayscale pixels and prepends a
classifier token internally.  These checkpoints were trained to length 1,024,
so this adapter regenerates the nonpersistent RoPE frequency buffer at length
1,025.  No learned parameter is added or resized.  This is therefore an
external paired adaptation, not an official LRA leaderboard entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoConfig, AutoModelForMaskedLM, get_cosine_schedule_with_warmup


EXPECTED_PRETRAIN_PARAMETERS = 99_985_152
EXPECTED_PRETRAIN_STEPS = 84_000
EXPECTED_PRETRAIN_TOKEN_POSITIONS = 1_376_256_000
OFFICIAL_LRA_REPOSITORY = "https://github.com/google-research/long-range-arena"
OFFICIAL_LRA_COMMIT = "cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8"
NATIVE_CONTEXT = 1024
PIXEL_SEQUENCE_LENGTH = 1024
MODEL_SEQUENCE_LENGTH = 1025
CLS_TOKEN_ID = 101
PIXEL_TOKEN_OFFSET = 1000
EXPECTED_DATA_SHA256 = {
    "train_pixels.npy": "6a9ae773a7fcbdf2cc4bd69729c10c8a6b8f6ca84e03e9ad942d7ee06ea8837b",
    "train_labels.npy": "1f58df08ea744abe109d461a3e40dd0e93ebaae2c14d9c3416c128ac51ad569a",
    "test_pixels.npy": "d45c7d4484cbba486a2d47e2b73aceabcda53f62e6c027140e9b0770a17cd75e",
    "test_labels.npy": "19004b019bd095265b4ce6637c9ff979c66abcbc8e658a665f791b487cdea5eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=("multispace-flash", "real-flash"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--backbone-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(modules: dict[str, nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        for tensor_name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{module_name}.{tensor_name}\0".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("LRA model execution requires CUDA; CPU execution is forbidden")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name:
        raise RuntimeError(f"LRA model execution requires an A100, got {name!r}")
    return {
        "name": name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda": torch.version.cuda,
    }


def validate_pretraining(model_path: Path, variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = model_path / "training_summary.json"
    config_path = model_path / "config.json"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint export: {model_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    expected_space = "multispace" if variant == "multispace-flash" else "real"
    checks = {
        "completed_schedule": summary.get("train/completed_schedule") is True,
        "optimizer_steps": int(summary.get("optimizer_steps", -1)) == EXPECTED_PRETRAIN_STEPS,
        "training_tokens": int(summary.get("training_tokens", -1)) == EXPECTED_PRETRAIN_TOKEN_POSITIONS,
        "attention_space": raw.get("attention_space") == expected_space,
        "attention_backend": raw.get("attention_backend") == "flash",
        "native_context": int(raw.get("max_length", -1)) == NATIVE_CONTEXT,
        "rope": raw.get("rope") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"pretraining fairness guard failed for {variant}: {failed}")
    return {
        "checks": checks,
        "optimizer_steps": int(summary["optimizer_steps"]),
        "training_tokens": int(summary["training_tokens"]),
        "attention_space": raw["attention_space"],
        "attention_backend": raw["attention_backend"],
        "native_max_length": int(raw["max_length"]),
    }, raw


def validate_data(data_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    digest = hashlib.sha256()
    for filename, expected in EXPECTED_DATA_SHA256.items():
        path = data_root / filename
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"data checksum mismatch for {path}: {observed} != {expected}")
        arrays[filename.removesuffix(".npy")] = np.load(path, mmap_mode="r", allow_pickle=False)
        digest.update(f"{filename}\0{observed}\n".encode())
    if arrays["train_pixels"].shape != (50_000, PIXEL_SEQUENCE_LENGTH):
        raise RuntimeError(f"unexpected train pixel shape: {arrays['train_pixels'].shape}")
    if arrays["test_pixels"].shape != (10_000, PIXEL_SEQUENCE_LENGTH):
        raise RuntimeError(f"unexpected test pixel shape: {arrays['test_pixels'].shape}")
    if arrays["train_labels"].shape != (50_000,) or arrays["test_labels"].shape != (10_000,):
        raise RuntimeError("unexpected label array shape")
    return arrays, manifest, digest.hexdigest()


class PixelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, pixels: np.ndarray, labels: np.ndarray, start: int, stop: int):
        self.pixels = pixels
        self.labels = labels
        self.start = start
        self.stop = stop

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = self.start + index
        # Offset gives all 256 pixel values distinct, non-padding vocabulary IDs.
        pixel_ids = torch.tensor(
            np.asarray(self.pixels[source_index], dtype=np.int64), dtype=torch.long
        ).add_(PIXEL_TOKEN_OFFSET)
        input_ids = torch.empty(MODEL_SEQUENCE_LENGTH, dtype=torch.long)
        input_ids[0] = CLS_TOKEN_ID
        input_ids[1:] = pixel_ids
        label = torch.tensor(int(self.labels[source_index]), dtype=torch.long)
        return input_ids, label


class FixedOrderSampler(Sampler[int]):
    def __init__(self, size: int, seed: int):
        order = np.arange(size, dtype=np.int64)
        np.random.default_rng(seed).shuffle(order)
        self.order = order.tolist()
        self.sha256 = hashlib.sha256(order.tobytes()).hexdigest()

    def __iter__(self) -> Iterator[int]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


class LRAClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone(input_ids)
        pooled = encoded[:, 0, :]
        return self.classifier(F.relu(self.dense(pooled)))


def initialize_head(model: LRAClassifier, seed: int) -> str:
    seed_everything(seed)
    for module in (model.dense, model.classifier):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.zeros_(module.bias)
    fingerprint = state_sha256({"dense": model.dense, "classifier": model.classifier})
    seed_everything(seed)
    return fingerprint


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    sampler: Sampler[int] | None,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        # Train/validation/test loaders coexist in this script.  Nonpersistent
        # workers prevent three worker pools from oversubscribing the allocation.
        persistent_workers=False,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    loss_sum = 0.0
    started = time.perf_counter()
    for input_ids, target in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if input_ids.shape[1] != MODEL_SEQUENCE_LENGTH:
            raise RuntimeError(f"LRA sequence was changed: {tuple(input_ids.shape)}")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.float(), target, reduction="sum")
        loss_sum += float(loss)
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(target.cpu().tolist())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "examples": len(labels),
        "loss": loss_sum / len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(10))).tolist(),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(labels) / elapsed,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def save_state(path: Path, model: nn.Module) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.micro_batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("epochs, micro batch, and accumulation must be positive")
    gpu = validate_gpu()
    seed_everything(args.seed)
    model_path = args.model.resolve()
    pretraining, raw_config = validate_pretraining(model_path, args.variant)
    arrays, data_manifest, data_fingerprint = validate_data(args.data_root.resolve())

    train_stop = min(45_000, args.max_train_examples or 45_000)
    valid_stop = min(50_000, 45_000 + (args.max_eval_examples or 5_000))
    test_stop = min(10_000, args.max_eval_examples or 10_000)
    train_dataset = PixelDataset(arrays["train_pixels"], arrays["train_labels"], 0, train_stop)
    valid_dataset = PixelDataset(
        arrays["train_pixels"], arrays["train_labels"], 45_000, valid_stop
    )
    test_dataset = PixelDataset(arrays["test_pixels"], arrays["test_labels"], 0, test_stop)

    # Extend only the nonpersistent analytical RoPE buffer.  Loading with this
    # config changes no checkpoint tensor and adds no trainable parameters.
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not config.rope or int(config.max_length) != NATIVE_CONTEXT:
        raise RuntimeError("the one-position adaptation requires native RoPE length 1024")
    config.max_length = MODEL_SEQUENCE_LENGTH
    loaded = AutoModelForMaskedLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    loaded_parameters = sum(parameter.numel() for parameter in loaded.parameters())
    if loaded_parameters != EXPECTED_PRETRAIN_PARAMETERS:
        raise RuntimeError(
            f"checkpoint parameter guard failed: {loaded_parameters:,} != "
            f"{EXPECTED_PRETRAIN_PARAMETERS:,}"
        )
    backbone = loaded.model
    del loaded
    backbone_parameters = sum(parameter.numel() for parameter in backbone.parameters())
    if backbone_parameters != EXPECTED_PRETRAIN_PARAMETERS:
        raise RuntimeError(
            f"backbone parameter guard failed: {backbone_parameters:,} != "
            f"{EXPECTED_PRETRAIN_PARAMETERS:,}"
        )
    if config.attention_backend != "flash" or config.attention_space != pretraining["attention_space"]:
        raise RuntimeError("attention implementation changed while loading LRA adapter")

    model = LRAClassifier(backbone, int(config.hidden_size), 10)
    head_initialization_sha256 = initialize_head(model, args.seed)
    head_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("dense.") or name.startswith("classifier.")
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if total_parameters - head_parameters != EXPECTED_PRETRAIN_PARAMETERS:
        raise RuntimeError("task head altered the exact checkpoint backbone parameter count")

    device = torch.device("cuda:0")
    model.to(device)
    valid_loader = make_loader(
        valid_dataset, args.eval_batch_size, sampler=None, num_workers=args.num_workers
    )
    test_loader = make_loader(
        test_dataset, args.eval_batch_size, sampler=None, num_workers=args.num_workers
    )
    batches_per_epoch = math.ceil(len(train_dataset) / args.micro_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = min(total_updates, round(updates_per_epoch * args.warmup_epochs))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": list(model.dense.parameters()) + list(model.classifier.parameters()),
                "lr": args.head_learning_rate,
                "weight_decay": args.weight_decay,
            },
        ],
        betas=(0.9, 0.98),
        eps=1.0e-9,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    run_dir = args.output_dir.resolve() / "cifar10" / args.variant / f"seed-{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pt"
    best_accuracy = -1.0
    best_epoch = -1
    optimizer_updates = 0
    history: list[dict[str, Any]] = []
    data_order_sha256: list[str] = []
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        sampler = FixedOrderSampler(len(train_dataset), args.seed + epoch - 1)
        data_order_sha256.append(sampler.sha256)
        train_loader = make_loader(
            train_dataset,
            args.micro_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        examples = 0
        for batch_index, (input_ids, target) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            group_offset = batch_index % args.gradient_accumulation
            group_remaining = len(train_loader) - (batch_index - group_offset)
            group_size = min(args.gradient_accumulation, group_remaining)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                raw_loss = F.cross_entropy(logits.float(), target)
                loss = raw_loss / group_size
            loss.backward()
            loss_sum += float(raw_loss.detach()) * target.numel()
            examples += target.numel()
            if group_offset + 1 == group_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
        validation = evaluate(model, valid_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_sum / examples,
            "optimizer_updates": optimizer_updates,
            "learning_rates": scheduler.get_last_lr(),
            "data_order_sha256": sampler.sha256,
            "validation": validation,
        }
        history.append(epoch_record)
        atomic_json(run_dir / "history.json", {"epochs": history})
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            save_state(best_path, model)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)

    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
    test = evaluate(model, test_loader, device)
    report = {
        "schema_version": 1,
        "benchmark": "Long Range Arena",
        "task": "Image / sequential grayscale CIFAR-10",
        "reporting_class": "external paired pretrained adaptation; not leaderboard-comparable",
        "official_lra": {
            "repository": OFFICIAL_LRA_REPOSITORY,
            "commit": OFFICIAL_LRA_COMMIT,
            "image_pipeline_reference": "lra_benchmarks/image/input_pipeline.py",
            "image_config_reference": (
                "lra_benchmarks/image/configs/cifar10/base_cifar10_config.py"
            ),
        },
        "dataset": {
            "manifest": data_manifest,
            "fingerprint_sha256": data_fingerprint,
            "rows": {
                "train": len(train_dataset),
                "validation": len(valid_dataset),
                "test": len(test_dataset),
            },
        },
        "model": {
            "variant": args.variant,
            "path": str(model_path),
            "checkpoint_parameters": backbone_parameters,
            "task_head_parameters": head_parameters,
            "total_finetune_parameters": total_parameters,
            "classifier_head_initialization_sha256": head_initialization_sha256,
            "pretraining": pretraining,
            "raw_config": {
                "max_length": raw_config["max_length"],
                "rope": raw_config["rope"],
                "attention_space": raw_config["attention_space"],
                "attention_backend": raw_config["attention_backend"],
            },
        },
        "adaptation": {
            "pixel_sequence_length": PIXEL_SEQUENCE_LENGTH,
            "model_sequence_length": MODEL_SEQUENCE_LENGTH,
            "pooling": "official CLS pooling",
            "classifier_token_id": CLS_TOKEN_ID,
            "pixel_token_mapping": f"token_id = uint8_pixel + {PIXEL_TOKEN_OFFSET}",
            "position_extension": (
                "regenerate analytical nonpersistent RoPE frequencies from 1024 to 1025; "
                "zero learned parameters"
            ),
            "truncation": "none",
            "padding": "none",
        },
        "gpu": gpu,
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "nominal_effective_batch_size": (
                args.micro_batch_size * args.gradient_accumulation
            ),
            "eval_batch_size": args.eval_batch_size,
            "backbone_learning_rate": args.backbone_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "betas": [0.9, 0.98],
            "epsilon": 1.0e-9,
            "scheduler": "linear warmup then cosine decay",
            "warmup_steps": warmup_steps,
            "total_updates": total_updates,
            "precision": "BF16 autocast with FP32 loss",
            "gradient_clip_norm": 1.0,
            "augmentation": "none",
        },
        "data_order_sha256": data_order_sha256,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "optimizer_updates": optimizer_updates,
        "training_seconds": training_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "test": test,
        "checkpoint": str(best_path),
        "history": history,
    }
    atomic_json(run_dir / "final_results.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
