#!/usr/bin/env python3
"""Paired full GLUE validation fine-tuning for the FineWeb-Edu checkpoints.

This intentionally implements the published GLUE metrics locally instead of
depending on a mutable metric download.  It uses the existing BabyLM
``sitecustomize`` compatibility hook to retain every learned tensor while
selecting NeoBERT's mathematically equivalent padded PyTorch attention path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast


EXPECTED_BASE_PARAMETERS = 99_985_152
DATASET_REPOSITORY = "nyu-mll/glue"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
METRIC_IMPLEMENTATION = "huggingface/evaluate metrics/glue/glue.py"
METRIC_REVISION = "e1a5d749a1772a37a8b68348d29f314a000d7907"
PROTOCOL_VERSION = "fineweb-glue-validation-v1"


@dataclass(frozen=True)
class TaskSpec:
    text_columns: tuple[str, ...]
    train_file: str
    validation_files: tuple[tuple[str, str], ...]
    num_outputs: int
    metric_kind: str
    labels: tuple[str, ...]


TASKS: dict[str, TaskSpec] = {
    "cola": TaskSpec(
        ("sentence",),
        "cola/train-00000-of-00001.parquet",
        (("validation", "cola/validation-00000-of-00001.parquet"),),
        2,
        "cola",
        ("unacceptable", "acceptable"),
    ),
    "sst2": TaskSpec(
        ("sentence",),
        "sst2/train-00000-of-00001.parquet",
        (("validation", "sst2/validation-00000-of-00001.parquet"),),
        2,
        "accuracy",
        ("negative", "positive"),
    ),
    "mrpc": TaskSpec(
        ("sentence1", "sentence2"),
        "mrpc/train-00000-of-00001.parquet",
        (("validation", "mrpc/validation-00000-of-00001.parquet"),),
        2,
        "acc_f1",
        ("not_equivalent", "equivalent"),
    ),
    "stsb": TaskSpec(
        ("sentence1", "sentence2"),
        "stsb/train-00000-of-00001.parquet",
        (("validation", "stsb/validation-00000-of-00001.parquet"),),
        1,
        "correlation",
        (),
    ),
    "qqp": TaskSpec(
        ("question1", "question2"),
        "qqp/train-00000-of-00001.parquet",
        (("validation", "qqp/validation-00000-of-00001.parquet"),),
        2,
        "acc_f1",
        ("not_duplicate", "duplicate"),
    ),
    "mnli": TaskSpec(
        ("premise", "hypothesis"),
        "mnli/train-00000-of-00001.parquet",
        (
            (
                "validation_matched",
                "mnli/validation_matched-00000-of-00001.parquet",
            ),
            (
                "validation_mismatched",
                "mnli/validation_mismatched-00000-of-00001.parquet",
            ),
        ),
        3,
        "accuracy",
        ("entailment", "neutral", "contradiction"),
    ),
    "qnli": TaskSpec(
        ("question", "sentence"),
        "qnli/train-00000-of-00001.parquet",
        (("validation", "qnli/validation-00000-of-00001.parquet"),),
        2,
        "accuracy",
        ("entailment", "not_entailment"),
    ),
    "rte": TaskSpec(
        ("sentence1", "sentence2"),
        "rte/train-00000-of-00001.parquet",
        (("validation", "rte/validation-00000-of-00001.parquet"),),
        2,
        "accuracy",
        ("entailment", "not_entailment"),
    ),
    "wnli": TaskSpec(
        ("sentence1", "sentence2"),
        "wnli/train-00000-of-00001.parquet",
        (("validation", "wnli/validation-00000-of-00001.parquet"),),
        2,
        "accuracy",
        ("not_entailment", "entailment"),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_a100() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("GLUE model execution requires CUDA")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name.upper():
        raise RuntimeError(f"GLUE model execution requires an A100; found {name}")
    return name


def configure_determinism(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set before Python starts"
        )
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class GlueHead(nn.Module):
    """BERT-style [CLS] classification/regression head."""

    def __init__(self, hidden_size: int, num_outputs: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_outputs)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = hidden[:, 0]
        pooled = self.dropout(pooled)
        pooled = torch.tanh(self.dense(pooled))
        pooled = self.dropout(pooled)
        return self.output(pooled)


class GlueModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: GlueHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.head(output.last_hidden_state)


def head_sha256(head: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(head.state_dict().items()):
        digest.update(f"{name}\0".encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_tokenizer(model_path: Path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True,
        )
    except (ValueError, OSError, KeyError):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_path,
            padding_side="right",
            local_files_only=True,
        )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("GLUE tokenizer has no pad token")
    return tokenizer


def load_model(
    model_path: Path,
    expected_space: str,
    seed: int,
    num_outputs: int,
    dropout: float,
) -> tuple[GlueModel, dict[str, Any]]:
    if os.environ.get("NEOBERT_BABYLM_FINETUNE_COMPAT") != "1":
        raise RuntimeError("NeoBERT padded-attention compatibility hook is disabled")

    # The shared compatibility hook restores RNG around architecture-specific
    # encoder construction.  Reset once more before the head as an explicit
    # pairing invariant independent of loader implementation details.
    set_seed(seed)
    encoder = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    base_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    if base_parameters != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(
            f"expected {EXPECTED_BASE_PARAMETERS:,} encoder parameters, "
            f"found {base_parameters:,}"
        )
    config = encoder.config
    if config.attention_space != expected_space:
        raise RuntimeError(
            f"expected attention_space={expected_space}; "
            f"found {config.attention_space}"
        )
    if config.attention_backend != "torch" or any(
        backend != "torch" for backend in config.attention_backends
    ):
        raise RuntimeError("padded GLUE fine-tuning must select torch attention")
    if next(encoder.parameters()).dtype != torch.float32:
        raise RuntimeError("GLUE encoder parameters must load in FP32")

    set_seed(seed)
    head = GlueHead(int(config.hidden_size), num_outputs, dropout)
    metadata = {
        "base_parameters": base_parameters,
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "head_initialization_sha256": head_sha256(head),
        "attention_space": config.attention_space,
        "checkpoint_attention_backend": "flash",
        "finetune_attention_backend": config.attention_backend,
        "parameter_dtype": str(next(encoder.parameters()).dtype),
    }
    return GlueModel(encoder, head), metadata


def verify_data_manifest(data_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = data_root / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"GLUE data manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError("GLUE data manifest has the wrong dataset revision")
    for record in manifest.get("files", []):
        path = data_root / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"missing or malformed GLUE file: {path}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"GLUE checksum mismatch: {path}")
    return manifest, sha256_file(manifest_path)


def load_parquet(path: Path) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(path)
    return Dataset.from_parquet(str(path))


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    text_columns: tuple[str, ...],
    max_length: int,
) -> Dataset:
    def tokenize(batch: dict[str, list[Any]], indices: list[int]):
        first = ["" if value is None else str(value) for value in batch[text_columns[0]]]
        second = None
        if len(text_columns) == 2:
            second = [
                "" if value is None else str(value)
                for value in batch[text_columns[1]]
            ]
        tokens = tokenizer(
            first,
            second,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        tokens["example_id"] = indices
        if "label" in batch:
            tokens["label"] = batch["label"]
        return tokens

    return dataset.map(
        tokenize,
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc=f"Tokenizing {dataset.num_rows} GLUE examples",
    )


def collate(rows: list[dict[str, Any]], tokenizer) -> dict[str, torch.Tensor]:
    encoded = [
        {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
        }
        for row in rows
    ]
    padded = tokenizer.pad(
        encoded,
        padding="longest",
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    padded["example_id"] = torch.tensor(
        [int(row["example_id"]) for row in rows],
        dtype=torch.long,
    )
    if "label" in rows[0]:
        labels = [row["label"] for row in rows]
        dtype = torch.float32 if any(isinstance(x, float) for x in labels) else torch.long
        padded["labels"] = torch.tensor(labels, dtype=dtype)
    return padded


def task_loss(logits: torch.Tensor, labels: torch.Tensor, regression: bool):
    if regression:
        return F.mse_loss(logits.squeeze(-1).float(), labels.float())
    return F.cross_entropy(logits.float(), labels.long())


def metrics_for(
    metric_kind: str,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    if metric_kind == "cola":
        return {
            "matthews_correlation": float(
                matthews_corrcoef(labels, predictions)
            )
        }
    if metric_kind == "correlation":
        return {
            "pearson": float(pearsonr(predictions, labels)[0]),
            "spearmanr": float(spearmanr(predictions, labels)[0]),
        }
    accuracy = float(np.mean(predictions == labels))
    if metric_kind == "acc_f1":
        return {
            "accuracy": accuracy,
            "f1": float(f1_score(labels, predictions)),
        }
    if metric_kind == "accuracy":
        return {"accuracy": accuracy}
    raise KeyError(metric_kind)


def split_score(metrics: dict[str, float]) -> float:
    if "matthews_correlation" in metrics:
        return metrics["matthews_correlation"]
    if "pearson" in metrics:
        return (metrics["pearson"] + metrics["spearmanr"]) / 2.0
    if "f1" in metrics:
        return (metrics["accuracy"] + metrics["f1"]) / 2.0
    return metrics["accuracy"]


@torch.no_grad()
def evaluate(
    model: GlueModel,
    dataset: Dataset,
    tokenizer,
    metric_kind: str,
    batch_size: int,
    regression: bool,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = collate(rows, tokenizer)
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        logits_parts.append(logits.float().cpu().numpy())
        labels_parts.append(batch["labels"].numpy())
        id_parts.append(batch["example_id"].numpy())
    logits_array = np.concatenate(logits_parts, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    example_ids = np.concatenate(id_parts, axis=0)
    predictions = (
        logits_array[:, 0] if regression else logits_array.argmax(axis=-1)
    )
    metrics = metrics_for(metric_kind, predictions, labels)
    return metrics, predictions, labels, example_ids, logits_array


@torch.no_grad()
def predict_unlabelled(
    model: GlueModel,
    dataset: Dataset,
    tokenizer,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = collate(rows, tokenizer)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                batch["input_ids"].cuda(non_blocking=True),
                batch["attention_mask"].cuda(non_blocking=True),
            )
        logits_parts.append(logits.float().cpu().numpy())
        id_parts.append(batch["example_id"].numpy())
    logits_array = np.concatenate(logits_parts, axis=0)
    example_ids = np.concatenate(id_parts, axis=0)
    return logits_array.argmax(axis=-1), example_ids, logits_array


def epoch_order(task: str, seed: int, epoch: int, size: int) -> list[int]:
    material = f"{PROTOCOL_VERSION}|{task}|{seed}|{epoch}".encode("utf-8")
    shuffle_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(shuffle_seed)
    return torch.randperm(size, generator=generator).tolist()


def order_sha256(order: list[int]) -> str:
    return hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()


def optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float,
):
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    warmup_steps = int(total_steps * warmup_ratio)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler, warmup_steps, len(decay), len(no_decay)


def hyperparameters(args: argparse.Namespace, task: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": task,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "classifier_dropout": args.classifier_dropout,
        "optimizer": "AdamW",
        "scheduler": "linear_decay_with_warmup",
        "precision": "BF16 autocast with FP32 parameters/optimizer",
        "pooling": "first_token_[CLS]",
        "checkpoint_selection": "none; fixed final epoch",
    }


def write_predictions(
    path: Path,
    predictions: np.ndarray,
    example_ids: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    arrays = {
        "predictions": predictions,
        "example_ids": example_ids,
        "logits": logits,
    }
    if labels is not None:
        arrays["labels"] = labels
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--expected-space",
        required=True,
        choices=("real", "multispace"),
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TASKS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--paired-preflight", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    device_name = require_a100()
    configure_determinism(args.seed)
    spec = TASKS[args.task]
    data_manifest, data_manifest_sha = verify_data_manifest(args.data_root)
    tokenizer = load_tokenizer(args.model)
    tokenizer_sha = sha256_file(args.model / "tokenizer.json")

    model, model_metadata = load_model(
        args.model,
        args.expected_space,
        args.seed,
        spec.num_outputs,
        args.classifier_dropout,
    )
    model.cuda()
    parameters_total = sum(parameter.numel() for parameter in model.parameters())
    model_metadata["total_trainable_parameters"] = parameters_total

    train_path = args.data_root / spec.train_file
    train = tokenize_dataset(
        load_parquet(train_path),
        tokenizer,
        spec.text_columns,
        args.max_length,
    )
    hparams = hyperparameters(args, args.task)
    hparams_sha = canonical_sha256(hparams)

    if args.preflight:
        rows = [train[index] for index in range(min(8, len(train)))]
        batch = collate(rows, tokenizer)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                batch["input_ids"].cuda(),
                batch["attention_mask"].cuda(),
            )
            loss = task_loss(
                logits,
                batch["labels"].cuda(),
                spec.num_outputs == 1,
            )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("GLUE preflight produced non-finite gradients")
        payload = {
            "status": "ok",
            "mode": "A100_forward_backward_preflight",
            "device": device_name,
            "model_path": str(args.model.resolve()),
            "model_weights_sha256": sha256_file(args.model / "model.safetensors"),
            "config_sha256": sha256_file(args.model / "config.json"),
            "tokenizer_sha256": tokenizer_sha,
            "variant": args.variant,
            "task": args.task,
            "seed": args.seed,
            "dataset_revision": DATASET_REVISION,
            "data_manifest_sha256": data_manifest_sha,
            "hyperparameters": hparams,
            "hyperparameters_sha256": hparams_sha,
            "loss": float(loss.detach()),
            "logits_shape": list(logits.shape),
            **model_metadata,
        }
        atomic_json(args.output_dir / "preflight.json", payload)
        print(json.dumps(payload, sort_keys=True))
        return

    if args.paired_preflight is None or not args.paired_preflight.is_file():
        raise FileNotFoundError("a completed paired preflight manifest is required")
    paired_preflight = json.loads(
        args.paired_preflight.read_text(encoding="utf-8")
    )
    variant_preflight = paired_preflight["variants"].get(args.variant)
    if variant_preflight is None:
        raise RuntimeError(f"variant is absent from paired preflight: {args.variant}")
    if variant_preflight["model_path"] != str(args.model.resolve()):
        raise RuntimeError("model path differs from the paired preflight")
    if variant_preflight["data_manifest_sha256"] != data_manifest_sha:
        raise RuntimeError("data manifest differs from the paired preflight")
    if variant_preflight["tokenizer_sha256"] != tokenizer_sha:
        raise RuntimeError("tokenizer differs from the paired preflight")

    validation_sets = {
        split: tokenize_dataset(
            load_parquet(args.data_root / relative_path),
            tokenizer,
            spec.text_columns,
            args.max_length,
        )
        for split, relative_path in spec.validation_files
    }
    steps_per_epoch = math.ceil(len(train) / args.train_batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer, scheduler, warmup_steps, decay_tensors, no_decay_tensors = (
        optimizer_and_scheduler(
            model,
            args.learning_rate,
            args.weight_decay,
            total_steps,
            args.warmup_ratio,
        )
    )

    epoch_records: list[dict[str, Any]] = []
    all_order_digest = hashlib.sha256()
    global_step = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    training_started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = epoch_order(args.task, args.seed, epoch, len(train))
        epoch_order_sha = order_sha256(order)
        all_order_digest.update(bytes.fromhex(epoch_order_sha))
        loss_sum = 0.0
        examples_seen = 0
        for start in range(0, len(order), args.train_batch_size):
            indices = order[start : start + args.train_batch_size]
            batch = collate([train[index] for index in indices], tokenizer)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    batch["input_ids"].cuda(non_blocking=True),
                    batch["attention_mask"].cuda(non_blocking=True),
                )
                loss = task_loss(
                    logits,
                    batch["labels"].cuda(non_blocking=True),
                    spec.num_outputs == 1,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch} step={global_step}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"non-finite gradient norm at epoch={epoch} step={global_step}"
                )
            optimizer.step()
            scheduler.step()
            count = len(indices)
            loss_sum += float(loss.detach()) * count
            examples_seen += count
            global_step += 1

        validation_metrics = {}
        for split, dataset in validation_sets.items():
            metrics, _, _, _, _ = evaluate(
                model,
                dataset,
                tokenizer,
                spec.metric_kind,
                args.evaluation_batch_size,
                spec.num_outputs == 1,
            )
            validation_metrics[split] = metrics
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": loss_sum / examples_seen,
                "examples_seen": examples_seen,
                "data_order_sha256": epoch_order_sha,
                "validation_metrics": validation_metrics,
            }
        )

    torch.cuda.synchronize()
    training_seconds = time.monotonic() - training_started
    final_validation = {}
    split_scores = []
    for split, dataset in validation_sets.items():
        metrics, predictions, labels, example_ids, logits = evaluate(
            model,
            dataset,
            tokenizer,
            spec.metric_kind,
            args.evaluation_batch_size,
            spec.num_outputs == 1,
        )
        predictions_file = args.output_dir / f"{split}_predictions.npz"
        predictions_sha = write_predictions(
            predictions_file,
            predictions,
            example_ids,
            logits,
            labels,
        )
        score = split_score(metrics)
        split_scores.append(score)
        final_validation[split] = {
            "examples": len(dataset),
            "metrics": metrics,
            "glue_split_score": score,
            "predictions_path": str(predictions_file.resolve()),
            "predictions_sha256": predictions_sha,
        }

    diagnostics = {}
    if args.task == "mnli":
        ax = tokenize_dataset(
            load_parquet(args.data_root / "ax/test-00000-of-00001.parquet"),
            tokenizer,
            spec.text_columns,
            args.max_length,
        )
        ax_predictions, ax_ids, ax_logits = predict_unlabelled(
            model,
            ax,
            tokenizer,
            args.evaluation_batch_size,
        )
        ax_file = args.output_dir / "ax_unscored_predictions.npz"
        ax_sha = write_predictions(ax_file, ax_predictions, ax_ids, ax_logits)
        diagnostics["ax"] = {
            "status": "unscored_hidden_test_labels",
            "examples": len(ax),
            "excluded_from_glue_aggregate": True,
            "predictions_path": str(ax_file.resolve()),
            "predictions_sha256": ax_sha,
        }

    completed_at = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "status": "complete",
        "evaluation": "GLUE public validation benchmark",
        "leaderboard_eligible": False,
        "leaderboard_note": (
            "Official GLUE test labels are private; these are local validation "
            "scores for a FineWeb-Edu-pretrained architecture comparison."
        ),
        "task": args.task,
        "seed": args.seed,
        "variant": args.variant,
        "model": {
            "path": str(args.model.resolve()),
            "model_weights_sha256": variant_preflight["model_weights_sha256"],
            "config_sha256": variant_preflight["config_sha256"],
            "tokenizer_sha256": tokenizer_sha,
            **model_metadata,
        },
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "manifest_path": str((args.data_root / "data_manifest.json").resolve()),
            "manifest_sha256": data_manifest_sha,
            "train_file": spec.train_file,
            "train_file_sha256": next(
                record["sha256"]
                for record in data_manifest["files"]
                if record["path"] == spec.train_file
            ),
            "train_examples": len(train),
            "validation_files": dict(spec.validation_files),
        },
        "metrics": {
            "implementation": METRIC_IMPLEMENTATION,
            "revision": METRIC_REVISION,
            "official_validation_metrics": True,
        },
        "labels": list(spec.labels),
        "hyperparameters": hparams,
        "hyperparameters_sha256": hparams_sha,
        "pairing": {
            "head_initialization_sha256": model_metadata[
                "head_initialization_sha256"
            ],
            "data_order_sha256": all_order_digest.hexdigest(),
            "epoch_data_order_sha256": [
                record["data_order_sha256"] for record in epoch_records
            ],
            "paired_preflight_path": str(args.paired_preflight.resolve()),
        },
        "optimization": {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "completed_steps": global_step,
            "warmup_steps": warmup_steps,
            "decay_parameter_tensors": decay_tensors,
            "no_decay_parameter_tensors": no_decay_tensors,
        },
        "epochs": epoch_records,
        "validation": final_validation,
        "glue_task_score": float(np.mean(split_scores)),
        "diagnostics": diagnostics,
        "hardware": {
            "device": device_name,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "timing": {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "training_seconds": training_seconds,
        },
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
