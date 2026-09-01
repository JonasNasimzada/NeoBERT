#!/usr/bin/env python3
"""Paired CLUTRR transfer evaluation for the matched FineWeb-Edu models.

The script intentionally performs every model operation on CUDA.  Dataset
parquet files are downloaded separately and pinned by SHA256 so an evaluation
never silently changes when a Hub branch moves.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


EXPECTED_PRETRAIN_PARAMETERS = 99_985_152
EXPECTED_PRETRAIN_STEPS = 84_000
EXPECTED_PRETRAIN_TOKEN_POSITIONS = 1_376_256_000
DATASET_REVISION = "6e5a7ac1425b8009b81ebc78c4f2b4d8ccd77722"
EXPECTED_SHA256 = {
    "gen_train23_test2to10": {
        "train": "6bcc82157d3b43684168bebaa9b0f7354165725b6b085c70a3f6a23ad5910c6b",
        "validation": "2488a8ac8063d8823d1cb86cace08c9d6cecaf1c96c8329717e9263fc51be5e9",
        "test": "6ef8a4b3e49a8cc6ecdf2d5a8f2bf4dbb6e7145123d4aae07da8f9947e3e3adc",
    },
    "rob_train_clean_23_test_all_23": {
        "train": "1c91de9a4c8a530f35926d6fed89cf3bdf6098e010bf8b3908534ac3bee4ea4f",
        "validation": "52fda18eb637c62db291ad4cc1276954af87abc46efa1f3d3101ed433190f479",
        "test": "de4297070768d0491307937ec0ee893b61ed2e0df0376fede74bfa57d31d1495",
    },
    "rob_train_disc_23_test_all_23": {
        "train": "98e61b9a63721fe62a8d0742fae7c124bb2b5447770bdba4a594a2f669bd9914",
        "validation": "b84d2aac4dc2be6146cd18aea37aecd558d90923f2ff15e74cec3498ad22b130",
        "test": "8ae0e4ee175e695118e55279a326c96a4fff1c7ef8e3206c488d1b4c5ce96159",
    },
    "rob_train_irr_23_test_all_23": {
        "train": "fcb5adabe5c861b37b84c8e26ee6d978c9530740242f7b2fd282ea102840dc3c",
        "validation": "8d3f10674bedac5cb4ae15f2eb2506c9b2deb7d6e4e0620c8185f9e74f3eb75f",
        "test": "97b65f774025c42fd36a90226dc38ebb93ddc16a2478d81659d4d0e3676fb897",
    },
    "rob_train_sup_23_test_all_23": {
        "train": "6e76f574a2663d78592bca748e1e55057ff4b35e3f7d021801c6f268d70889ff",
        "validation": "5a226c3696f5e1642e7b0ccce68377b4e577ec37d29f1297c6eea50d5ce2f6f9",
        "test": "790c8a75f1b785091feeb6541b4b20808e27b97e57295e64de43ba0faea31aa8",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=("multispace-flash", "real-flash"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", default="gen_train23_test2to10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-proportion", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--skip-checksum", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(modules: dict[str, torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        for tensor_name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{module_name}.{tensor_name}\0".encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CLUTRR evaluation requires CUDA; CPU execution is forbidden")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "A100" not in name:
        raise RuntimeError(f"CLUTRR evaluation requires an A100, got {name!r}")
    return {"name": name, "capability": list(capability), "torch_cuda": torch.version.cuda}


def validate_pretraining(model_path: Path, variant: str) -> dict[str, Any]:
    summary_path = model_path / "training_summary.json"
    config_path = model_path / "config.json"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"incomplete final export: {model_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_space = "multispace" if variant == "multispace-flash" else "real"
    checks = {
        "completed_schedule": summary.get("train/completed_schedule") is True,
        "optimizer_steps": int(summary.get("optimizer_steps", -1)) == EXPECTED_PRETRAIN_STEPS,
        "training_tokens": int(summary.get("training_tokens", -1)) == EXPECTED_PRETRAIN_TOKEN_POSITIONS,
        "attention_space": raw_config.get("attention_space") == expected_space,
        "attention_backend": raw_config.get("attention_backend") == "flash",
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"pretraining fairness guard failed for {variant}: {failed}")
    return {
        "checks": checks,
        "optimizer_steps": summary["optimizer_steps"],
        "training_tokens": summary["training_tokens"],
        "attention_space": raw_config["attention_space"],
        "attention_backend": raw_config["attention_backend"],
        "hidden_size": raw_config["hidden_size"],
        "num_hidden_layers": raw_config["num_hidden_layers"],
        "num_attention_heads": raw_config["num_attention_heads"],
        "intermediate_size": raw_config["intermediate_size"],
    }


def relation_question(raw_query: str) -> str:
    pair = ast.literal_eval(raw_query)
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError(f"invalid CLUTRR query: {raw_query!r}")
    source, target = pair
    return f"What is {target}'s family relation to {source}?"


def task_length(task_name: str) -> int:
    return int(task_name.rsplit(".", 1)[1])


def prepare_split(dataset, tokenizer, max_length: int, max_examples: int | None):
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    def tokenize(batch):
        questions = [relation_question(query) for query in batch["query"]]
        encoded = tokenizer(
            batch["story"],
            questions,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_length=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "length": encoded["length"],
            "labels": batch["target"],
            "task_name_eval": batch["task_name"],
            "task_length_eval": [task_length(name) for name in batch["task_name"]],
        }

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Tokenizing pinned CLUTRR data",
    )


class Collator:
    def __call__(self, examples):
        lengths = {len(row["input_ids"]) for row in examples}
        if len(lengths) != 1:
            raise RuntimeError(f"Flash CLUTRR batches must be padding-free; got lengths {lengths}")
        batch = {
            "input_ids": torch.tensor(
                [row["input_ids"] for row in examples], dtype=torch.long
            )
        }
        batch["labels"] = torch.tensor([row["labels"] for row in examples], dtype=torch.long)
        batch["task_lengths"] = torch.tensor(
            [row["task_length_eval"] for row in examples], dtype=torch.long
        )
        batch["task_names"] = [row["task_name_eval"] for row in examples]
        batch["unpadded_lengths"] = torch.tensor([row["length"] for row in examples], dtype=torch.long)
        return batch


def label_mapping(datasets: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for dataset in datasets.values():
        for target, text in zip(dataset["target"], dataset["target_text"]):
            target = int(target)
            if target in mapping and mapping[target] != text:
                raise RuntimeError(f"CLUTRR label {target} maps to multiple relations")
            mapping[target] = text
    expected = list(range(max(mapping) + 1))
    if sorted(mapping) != expected:
        raise RuntimeError(f"CLUTRR labels are not contiguous: {sorted(mapping)}")
    train_labels = set(int(value) for value in datasets["train"]["target"])
    unseen_labels = sorted(set(mapping) - train_labels)
    if unseen_labels:
        raise RuntimeError(f"test-only relation labels cannot be learned: {unseen_labels}")
    return mapping


class ExactLengthBatchSampler(Sampler[list[int]]):
    """Deterministic batches with no padding, required by strict FlashAttention."""

    def __init__(self, dataset, batch_size: int, shuffle: bool, seed: int):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.buckets: dict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(dataset["length"]):
            self.buckets[int(length)].append(index)

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.buckets.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for length in sorted(self.buckets):
            indices = list(self.buckets[length])
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(
                indices[offset : offset + self.batch_size]
                for offset in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches


def dataloader(dataset, batch_size: int, shuffle: bool, seed: int):
    return DataLoader(
        dataset,
        batch_sampler=ExactLengthBatchSampler(dataset, batch_size, shuffle, seed),
        collate_fn=Collator(),
        num_workers=0,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(model, loader, device: torch.device, max_length: int) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    task_names: list[str] = []
    task_lengths: list[int] = []
    losses: list[float] = []
    examples = 0
    truncated = 0
    started = time.perf_counter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, return_dict=True).logits
            loss = F.cross_entropy(logits.float(), target, reduction="sum")
        predictions.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(target.cpu().tolist())
        task_names.extend(batch["task_names"])
        task_lengths.extend(batch["task_lengths"].tolist())
        losses.append(float(loss.item()))
        examples += target.numel()
        # The tokenizer reports length after truncation, so equality is an
        # intentionally conservative upper bound rather than an exact count.
        truncated += int((batch["unpadded_lengths"] >= max_length).sum())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "examples": examples,
        "loss": sum(losses) / examples,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed,
        "truncated_examples_upper_bound": truncated,
    }
    by_task: dict[str, dict[str, float | int]] = {}
    indices_by_task: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(task_names):
        indices_by_task[name].append(index)
    for name, indices in sorted(indices_by_task.items()):
        gold = [labels[index] for index in indices]
        pred = [predictions[index] for index in indices]
        by_task[name] = {
            "examples": len(indices),
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
        }
    result["by_task"] = by_task

    seen = [index for index, length in enumerate(task_lengths) if length <= 3]
    unseen = [index for index, length in enumerate(task_lengths) if length >= 4]
    for group_name, indices in (("seen_k2_k3", seen), ("unseen_k4_k10", unseen)):
        if indices:
            gold = [labels[index] for index in indices]
            pred = [predictions[index] for index in indices]
            result[group_name] = {
                "examples": len(indices),
                "accuracy": accuracy_score(gold, pred),
                "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
            }
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation < 1:
        raise ValueError("gradient accumulation must be positive")
    if args.batch_size % args.gradient_accumulation:
        raise ValueError("batch size must be divisible by gradient accumulation")
    gpu = validate_gpu()
    seed_everything(args.seed)
    pretraining = validate_pretraining(args.model.resolve(), args.variant)

    data_dir = args.data_root.resolve() / args.config
    paths = {split: data_dir / f"{split}.parquet" for split in ("train", "validation", "test")}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = EXPECTED_SHA256.get(args.config, {}).get(split)
        if expected and not args.skip_checksum:
            observed = sha256(path)
            if observed != expected:
                raise RuntimeError(f"checksum mismatch for {path}: {observed} != {expected}")

    raw = {
        split: load_dataset("parquet", data_files=str(path), split="train")
        for split, path in paths.items()
    }
    mapping = label_mapping(raw)
    num_labels = len(mapping)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    tokenized = {
        "train": prepare_split(raw["train"], tokenizer, args.max_length, args.max_train_examples),
        "validation": prepare_split(raw["validation"], tokenizer, args.max_length, args.max_eval_examples),
        "test": prepare_split(raw["test"], tokenizer, args.max_length, args.max_eval_examples),
    }

    # Reset immediately before classifier creation so both variants receive the
    # same head initialization for a given transfer seed.
    seed_everything(args.seed)
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        num_labels=num_labels,
        id2label={index: label for index, label in mapping.items()},
        label2id={label: index for index, label in mapping.items()},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )
    # ``from_pretrained`` constructs the variant-specific encoder tree before
    # it initializes the downstream head.  Those trees consume RNG differently,
    # so reset after loading and explicitly create one paired head initialization.
    seed_everything(args.seed)
    model._init_weights(model.dense)
    model._init_weights(model.classifier)
    head_initialization_sha256 = state_sha256(
        {"dense": model.dense, "classifier": model.classifier}
    )
    # Start shuffling and classifier dropout from the same RNG state as well.
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    model.to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    head_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("dense.") or name.startswith("classifier.")
    )
    encoder_parameters = trainable_parameters - head_parameters
    if trainable_parameters <= 0:
        raise RuntimeError("model has no trainable parameters")
    if encoder_parameters != EXPECTED_PRETRAIN_PARAMETERS:
        raise RuntimeError(
            f"parameter guard failed: encoder={encoder_parameters:,}, "
            f"expected={EXPECTED_PRETRAIN_PARAMETERS:,}"
        )
    if pretraining["attention_space"] == "multispace" and config.attention_space != "multispace":
        raise RuntimeError("multispace checkpoint changed attention space during loading")
    if pretraining["attention_space"] == "real" and config.attention_space != "real":
        raise RuntimeError("real checkpoint changed attention space during loading")

    micro_batch = args.batch_size // args.gradient_accumulation
    train_loader = dataloader(tokenized["train"], micro_batch, True, args.seed)
    valid_loader = dataloader(tokenized["validation"], args.eval_batch_size, False, args.seed)
    test_loader = dataloader(tokenized["test"], args.eval_batch_size, False, args.seed)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = round(total_updates * args.warmup_proportion)

    decay_names = ("bias", "norm.weight", "layer_norm.weight")
    grouped = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not any(token in name.lower() for token in decay_names)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and any(token in name.lower() for token in decay_names)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    run_dir = args.output_dir.resolve() / args.config / args.variant / f"seed-{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = -1
    optimizer_updates = 0
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        examples = 0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, labels=labels, return_dict=True)
                loss = output.loss / args.gradient_accumulation
            loss.backward()
            loss_sum += float(output.loss.detach()) * labels.numel()
            examples += labels.numel()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
        validation = evaluate(model, valid_loader, device, args.max_length)
        epoch_result = {
            "epoch": epoch,
            "train_loss": loss_sum / examples,
            "optimizer_updates": optimizer_updates,
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation,
        }
        history.append(epoch_result)
        atomic_json(run_dir / "history.json", {"epochs": history})
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, checkpoint_path)
        print(json.dumps(epoch_result, sort_keys=True), flush=True)

    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, device, args.max_length)
    final = {
        "schema_version": 1,
        "benchmark": "CLUTRR",
        "dataset": {
            "repo": "CLUTRR/v1",
            "revision": DATASET_REVISION,
            "config": args.config,
            "paths": {key: str(value) for key, value in paths.items()},
            "sha256": {key: sha256(value) for key, value in paths.items()},
            "rows": {key: len(value) for key, value in raw.items()},
            "label_mapping": {str(key): value for key, value in mapping.items()},
        },
        "model": {
            "variant": args.variant,
            "path": str(args.model.resolve()),
            "pretrain_parameters": EXPECTED_PRETRAIN_PARAMETERS,
            "finetune_parameters": trainable_parameters,
            "classifier_head_parameters": head_parameters,
            "classifier_head_initialization_sha256": head_initialization_sha256,
            "encoder_parameters": encoder_parameters,
            "pretraining": pretraining,
        },
        "gpu": gpu,
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "micro_batch_size": micro_batch,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_proportion": args.warmup_proportion,
            "warmup_steps": warmup_steps,
            "max_length": args.max_length,
            "total_updates": total_updates,
            "batching": "exact_token_length_no_padding",
        },
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "training_seconds": training_seconds,
        "optimizer_updates": optimizer_updates,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "test": test,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    atomic_json(run_dir / "final_results.json", final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
