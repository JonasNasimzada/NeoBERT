#!/usr/bin/env python3
"""Full paired SuperGLUE validation evaluation for the exact 100M models.

The eight main tasks use their official validation metrics.  AX-b and AX-g
are evaluated as labeled diagnostics through the RTE head, but never enter the
primary aggregate.  Main-task test labels are hidden, so this protocol is an
external validation benchmark rather than a leaderboard submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

from evaluation_pipeline.finetune.classifier_model import ClassifierHead


EXPECTED_BASE_PARAMETERS = 99_985_152
DATASET_ID = "aps/super_glue"
DATASET_REVISION = "3de24cf8022e94f4ee4b9d55a6f539891524d646"
METRIC_IMPLEMENTATION = {
    "repository": "https://github.com/huggingface/datasets",
    "revision": "06c3ffb8d068b6307b247164b10f7c7311cefed4",
    "tag": "2.14.6",
    "super_glue.py_sha256": "dbcb97530334cd0798b432a4d02b31f1642a313b7687846b972da03eeaead9a3",
    "record_evaluation.py_sha256": "4bda909c08d5c9e21ce2d41f04ce5415c5d341976c22119da2905a37d9fca6cf",
}
TASKS = ("boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc")
EXPECTED_MODEL_SHA256 = {
    "multispace": "f28d321a233d6305d4983ba77d3a54cd54974db3343820af50292b3bcc39d94f",
    "real": "4699f5ae13b5b296258b4a3d120a7b7386450392d4f97ea9a894a56ad04fe992",
}
DEFAULT_EPOCHS = {
    "boolq": 5,
    "cb": 20,
    "copa": 20,
    "multirc": 5,
    "record": 2,
    "rte": 10,
    "wic": 10,
    "wsc": 20,
}
NUM_LABELS = {
    "boolq": 2,
    "cb": 3,
    "copa": 1,
    "multirc": 2,
    "record": 2,
    "rte": 2,
    "wic": 2,
    "wsc": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_a100() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SuperGLUE model execution must run on CUDA")
    device = torch.cuda.get_device_name(0)
    if "A100" not in device.upper():
        raise RuntimeError(f"SuperGLUE requires an A100; found {device}")
    return {
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_tokenizer(path: Path):
    kwargs = {"trust_remote_code": True, "padding_side": "left"}
    try:
        tokenizer = AutoProcessor.from_pretrained(str(path), **kwargs)
    except (ValueError, OSError, KeyError):
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(path), **kwargs)
        except (ValueError, OSError, KeyError):
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                str(path), padding_side="left"
            )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("SuperGLUE requires the checkpoint's fast tokenizer")
    return tokenizer


def validate_data(
    data_root: Path, manifest_path: Path, task: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET_ID:
        raise RuntimeError(f"unexpected dataset id in {manifest_path}")
    if manifest.get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError(f"unexpected dataset revision in {manifest_path}")
    if manifest.get("metric_implementation") != METRIC_IMPLEMENTATION:
        raise RuntimeError(f"unexpected metric implementation in {manifest_path}")
    data_task = "wsc.fixed" if task == "wsc" else task
    required = [f"{data_task}/train.parquet", f"{data_task}/validation.parquet"]
    if task == "rte":
        required.extend(("axb/test.parquet", "axg/test.parquet"))
    verified: dict[str, Any] = {}
    for relative in required:
        entry = manifest["files"].get(relative)
        if not entry:
            raise RuntimeError(f"data manifest is missing {relative}")
        path = (data_root / relative).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"SuperGLUE file is missing: {path}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise RuntimeError(
                f"hash mismatch for {relative}: expected {entry['sha256']}, found {digest}"
            )
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != int(entry["rows"]):
            raise RuntimeError(
                f"row-count mismatch for {relative}: expected {entry['rows']}, found {rows}"
            )
        verified[relative] = {
            "path": str(path),
            "rows": rows,
            "sha256": digest,
        }
    return manifest, verified


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, min(limit, table.num_rows))
    return table.to_pylist()


def json_id(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: int(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def mark_char_span(text: str, start: int, end: int) -> str:
    if not (0 <= start < end <= len(text)):
        return text
    return f"{text[:start]} [unused0] {text[start:end]} [unused1] {text[end:]}"


def mark_wsc(row: dict[str, Any]) -> str:
    tokens = row["text"].split()
    annotations = [
        (int(row["span1_index"]), len(row["span1_text"].split()), "[unused0]", "[unused1]"),
        (int(row["span2_index"]), len(row["span2_text"].split()), "[unused2]", "[unused3]"),
    ]
    for start, width, left, right in sorted(annotations, reverse=True):
        if start < 0 or start >= len(tokens):
            raise RuntimeError(f"invalid WSC span index {start} for {len(tokens)} tokens")
        tokens.insert(min(start + width, len(tokens)), right)
        tokens.insert(start, left)
    return " ".join(tokens)


def sequence_examples(task: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        label = int(row["label"])
        if label < 0:
            raise RuntimeError(f"{task} contains an unavailable label")
        group = None
        if task == "boolq":
            text_a = f"question: {row['question']}"
            text_b = row["passage"]
        elif task in ("cb", "rte"):
            text_a, text_b = row["premise"], row["hypothesis"]
        elif task == "multirc":
            text_a = row["paragraph"]
            text_b = f"question: {row['question']} answer: {row['answer']}"
            group = [int(row["idx"]["paragraph"]), int(row["idx"]["question"])]
        elif task == "wic":
            sentence1 = mark_char_span(
                row["sentence1"], int(row["start1"]), int(row["end1"])
            )
            sentence2 = mark_char_span(
                row["sentence2"], int(row["start2"]), int(row["end2"])
            )
            text_a = f"word: {row['word']} sentence 1: {sentence1}"
            text_b = f"sentence 2: {sentence2}"
        elif task == "wsc":
            text_a = mark_wsc(row)
            text_b = f"candidate: {row['span1_text']} pronoun: {row['span2_text']}"
        else:
            raise ValueError(f"not a sequence task: {task}")
        examples.append(
            {
                "id": json_id(row["idx"]),
                "text_a": text_a,
                "text_b": text_b,
                "label": label,
                "group": group,
            }
        )
    return examples


def diagnostic_examples(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if name == "axb":
            first, second = row["sentence1"], row["sentence2"]
        elif name == "axg":
            first, second = row["premise"], row["hypothesis"]
        else:
            raise ValueError(name)
        examples.append(
            {
                "id": json_id(row["idx"]),
                "text_a": first,
                "text_b": second,
                "label": int(row["label"]),
                "group": None,
            }
        )
    return examples


def copa_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        relation = "cause" if row["question"] == "cause" else "effect"
        choices = [row["choice1"], row["choice2"]]
        paired = []
        for choice in choices:
            if relation == "cause":
                paired.append((row["premise"], f"because: {choice}"))
            else:
                paired.append((row["premise"], f"therefore: {choice}"))
        examples.append(
            {"id": json_id(row["idx"]), "pairs": paired, "label": int(row["label"])}
        )
    return examples


class ListDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class SequenceCollator:
    def __init__(self, tokenizer, task: str, max_length: int):
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if self.task == "boolq":
            truncation: str | bool = "only_second"
        elif self.task == "multirc":
            truncation = "only_first"
        else:
            truncation = True
        encoded = self.tokenizer(
            [item["text_a"] for item in batch],
            [item["text_b"] for item in batch],
            padding=True,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "ids": [item["id"] for item in batch],
            "groups": [item.get("group") for item in batch],
        }


class CopaCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        first = [pair[0] for item in batch for pair in item["pairs"]]
        second = [pair[1] for item in batch for pair in item["pairs"]]
        encoded = self.tokenizer(
            first,
            second,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_size, choices = len(batch), 2
        return {
            "input_ids": encoded.input_ids.view(batch_size, choices, -1),
            "attention_mask": encoded.attention_mask.view(batch_size, choices, -1),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "ids": [item["id"] for item in batch],
        }


def record_spans(row: dict[str, Any]) -> list[tuple[str, int, int]]:
    spans = row["entity_spans"]
    if isinstance(spans, dict):
        return [
            (str(text), int(start), int(end))
            for text, start, end in zip(
                spans["text"], spans["start"], spans["end"], strict=True
            )
        ]
    return [
        (str(item["text"]), int(item["start"]), int(item["end"]))
        for item in spans
    ]


def crop_around(text: str, start: int, end: int, budget: int) -> tuple[str, int, int]:
    center = (start + end) // 2
    crop_start = max(0, center - budget // 2)
    crop_end = min(len(text), crop_start + budget)
    crop_start = max(0, crop_end - budget)
    return text[crop_start:crop_end], start - crop_start, end - crop_start


def token_span(
    offsets: Sequence[Sequence[int]], sequence_ids: Sequence[int | None], start: int, end: int
) -> tuple[int, int]:
    candidates = [
        index
        for index, ((left, right), sequence_id) in enumerate(zip(offsets, sequence_ids))
        if sequence_id == 1 and right > start and left < end
    ]
    if not candidates:
        raise RuntimeError(f"answer span [{start}, {end}) was lost during tokenization")
    return candidates[0], candidates[-1]


class RecordTrainCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.char_budget = max_length * 3

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        crops: list[str] = []
        local_spans: list[tuple[int, int]] = []
        for row in batch:
            answers = set(row["answers"])
            gold = [span for span in record_spans(row) if span[0] in answers]
            if not gold:
                for answer in row["answers"]:
                    start = row["passage"].find(answer)
                    if start >= 0:
                        gold.append((answer, start, start + len(answer)))
            if not gold:
                raise RuntimeError(f"ReCoRD row {row['idx']} has no answer occurrence")
            choice = int(row["idx"]["query"]) % len(gold)
            _, start, end = gold[choice]
            crop, local_start, local_end = crop_around(
                row["passage"], start, end, self.char_budget
            )
            crops.append(crop)
            local_spans.append((local_start, local_end))
        encoded = self.tokenizer(
            [row["query"] for row in batch],
            crops,
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        starts, ends = [], []
        offsets = encoded.pop("offset_mapping")
        for index, (start, end) in enumerate(local_spans):
            left, right = token_span(
                offsets[index].tolist(), encoded.sequence_ids(index), start, end
            )
            starts.append(left)
            ends.append(right)
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "start_positions": torch.tensor(starts, dtype=torch.long),
            "end_positions": torch.tensor(ends, dtype=torch.long),
        }


def record_eval_features(rows: list[dict[str, Any]], max_length: int) -> list[dict[str, Any]]:
    char_budget = max_length * 3
    features = []
    for row_index, row in enumerate(rows):
        entity_to_index = {entity: index for index, entity in enumerate(row["entities"])}
        for text, start, end in record_spans(row):
            if text not in entity_to_index:
                continue
            crop, local_start, local_end = crop_around(
                row["passage"], start, end, char_budget
            )
            features.append(
                {
                    "row_index": row_index,
                    "entity_index": entity_to_index[text],
                    "query": row["query"],
                    "passage": crop,
                    "start": local_start,
                    "end": local_end,
                }
            )
    return features


class RecordEvalCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [item["query"] for item in batch],
            [item["passage"] for item in batch],
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        positions = []
        for index, item in enumerate(batch):
            positions.append(
                token_span(
                    offsets[index].tolist(),
                    encoded.sequence_ids(index),
                    int(item["start"]),
                    int(item["end"]),
                )
            )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "positions": positions,
            "row_indices": [int(item["row_index"]) for item in batch],
            "entity_indices": [int(item["entity_index"]) for item in batch],
        }


class SuperGlueModel(nn.Module):
    def __init__(self, base_model: Path, task: str):
        super().__init__()
        # sitecustomize replaces AutoModel.from_pretrained for local NeoBERT
        # exports and restores RNG after encoder construction.
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(str(base_model), trust_remote_code=True)
        hidden_size = int(self.encoder.config.hidden_size)
        head_config = SimpleNamespace(
            hidden_size=hidden_size,
            classifier_layer_norm_eps=1.0e-5,
            classifier_dropout=0.1,
            num_labels=NUM_LABELS[task],
        )
        self.head = ClassifierHead(head_config, hidden_size)
        self.task = task

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        if self.task == "copa":
            batch, choices, length = input_ids.shape
            hidden = self.encode(
                input_ids.reshape(batch * choices, length),
                attention_mask.reshape(batch * choices, length),
            )
            return self.head(hidden[:, -1]).reshape(batch, choices)
        hidden = self.encode(input_ids, attention_mask)
        if self.task == "record":
            logits = self.head(hidden)
            return logits[..., 0], logits[..., 1]
        return self.head(hidden[:, -1])


def head_digest(model: SuperGlueModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.head.state_dict().items()):
        digest.update(f"{name}\0".encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_model(model: SuperGlueModel, expected_space: str) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    if parameters != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(
            f"expected {EXPECTED_BASE_PARAMETERS:,} encoder parameters; found {parameters:,}"
        )
    config = model.encoder.config
    if config.attention_space != expected_space:
        raise RuntimeError(
            f"expected attention_space={expected_space}; found {config.attention_space}"
        )
    if config.attention_backend != "torch" or any(
        backend != "torch" for backend in config.attention_backends
    ):
        raise RuntimeError("SuperGLUE padded batches require paired torch attention")
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("SuperGLUE protocol requires FP32 model weights")
    return {
        "base_parameters": parameters,
        "total_parameters_with_task_head": sum(p.numel() for p in model.parameters()),
        "attention_space": config.attention_space,
        "checkpoint_attention_backend": "flash",
        "finetune_attention_backend": config.attention_backend,
        "dtype": str(next(model.parameters()).dtype),
    }


def move(batch: dict[str, Any], *names: str) -> list[torch.Tensor]:
    return [batch[name].cuda(non_blocking=True) for name in names]


def make_loader(
    rows: list[dict[str, Any]],
    collator,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ListDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )


def train(
    model: SuperGlueModel,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup: float,
    max_train_steps: int | None,
) -> dict[str, Any]:
    steps_per_epoch = len(loader)
    planned = epochs * steps_per_epoch
    total_steps = planned if max_train_steps is None else min(planned, max_train_steps)
    if total_steps <= 0:
        raise RuntimeError("training has no optimizer steps")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * warmup),
        num_training_steps=total_steps,
    )
    model.train()
    losses: list[float] = []
    started = time.perf_counter()
    steps = 0
    for _epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids, attention_mask = move(batch, "input_ids", "attention_mask")
            output = model(input_ids, attention_mask)
            if model.task == "record":
                starts, ends = move(batch, "start_positions", "end_positions")
                start_logits, end_logits = output
                loss = (F.cross_entropy(start_logits, starts) + F.cross_entropy(end_logits, ends)) / 2
            else:
                labels = batch["labels"].cuda(non_blocking=True)
                loss = F.cross_entropy(output, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {steps}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            steps += 1
            if steps >= total_steps:
                break
        if steps >= total_steps:
            break
    torch.cuda.synchronize()
    return {
        "optimizer_steps": steps,
        "planned_optimizer_steps": planned,
        "epochs": epochs,
        "mean_loss": sum(losses) / len(losses),
        "final_loss": losses[-1],
        "seconds": time.perf_counter() - started,
    }


def accuracy(gold: Sequence[int], pred: Sequence[int]) -> float:
    return sum(int(a == b) for a, b in zip(gold, pred, strict=True)) / len(gold)


def macro_f1(gold: Sequence[int], pred: Sequence[int], labels: Sequence[int]) -> float:
    values = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(gold, pred, strict=True))
        fp = sum(a != label and b == label for a, b in zip(gold, pred, strict=True))
        fn = sum(a == label and b != label for a, b in zip(gold, pred, strict=True))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(values) / len(values)


def binary_f1(gold: Sequence[int], pred: Sequence[int]) -> float:
    tp = sum(a == 1 and b == 1 for a, b in zip(gold, pred, strict=True))
    fp = sum(a == 0 and b == 1 for a, b in zip(gold, pred, strict=True))
    fn = sum(a == 1 and b == 0 for a, b in zip(gold, pred, strict=True))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def matthews_correlation(gold: Sequence[int], pred: Sequence[int]) -> float:
    tp = sum(a == 1 and b == 1 for a, b in zip(gold, pred, strict=True))
    tn = sum(a == 0 and b == 0 for a, b in zip(gold, pred, strict=True))
    fp = sum(a == 0 and b == 1 for a, b in zip(gold, pred, strict=True))
    fn = sum(a == 1 and b == 0 for a, b in zip(gold, pred, strict=True))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator


@torch.inference_mode()
def evaluate_classification(
    model: SuperGlueModel, loader: DataLoader, task: str
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    gold: list[int] = []
    pred: list[int] = []
    records: list[dict[str, Any]] = []
    question_groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for batch in loader:
        input_ids, attention_mask = move(batch, "input_ids", "attention_mask")
        logits = model(input_ids, attention_mask)
        outputs = logits.argmax(dim=-1).cpu().tolist()
        targets = batch["labels"].tolist()
        gold.extend(targets)
        pred.extend(outputs)
        for index, (target, output) in enumerate(zip(targets, outputs, strict=True)):
            record = {"id": batch["ids"][index], "gold": target, "prediction": output}
            if task == "multirc":
                group = tuple(batch["groups"][index])
                question_groups[group].append((target, output))
                record["question_group"] = list(group)
            records.append(record)
    torch.cuda.synchronize()
    metrics: dict[str, float]
    if task == "cb":
        metrics = {"accuracy": accuracy(gold, pred), "f1": macro_f1(gold, pred, (0, 1, 2))}
    elif task == "multirc":
        exact_match = sum(
            all(target == output for target, output in values)
            for values in question_groups.values()
        ) / len(question_groups)
        per_question_f1 = []
        for values in question_groups.values():
            question_gold = [target for target, _ in values]
            question_pred = [output for _, output in values]
            present_labels = sorted(set(question_gold) | set(question_pred))
            per_question_f1.append(
                macro_f1(question_gold, question_pred, present_labels)
            )
        metrics = {
            "f1_a": binary_f1(gold, pred),
            "f1_m": sum(per_question_f1) / len(per_question_f1),
            "exact_match": exact_match,
        }
    elif task == "axb":
        metrics = {"matthews_correlation": matthews_correlation(gold, pred)}
    else:
        metrics = {"accuracy": accuracy(gold, pred)}
    return metrics, records, {
        "rows": len(gold),
        "seconds": time.perf_counter() - started,
        "examples_per_second": len(gold) / (time.perf_counter() - started),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def record_exact(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def record_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@torch.inference_mode()
def evaluate_record(
    model: SuperGlueModel,
    rows: list[dict[str, Any]],
    loader: DataLoader,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    scores = [[-math.inf] * len(row["entities"]) for row in rows]
    feature_count = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for batch in loader:
        input_ids, attention_mask = move(batch, "input_ids", "attention_mask")
        start_logits, end_logits = model(input_ids, attention_mask)
        start_positions = torch.tensor(
            [position[0] for position in batch["positions"]],
            dtype=torch.long,
            device=start_logits.device,
        )
        end_positions = torch.tensor(
            [position[1] for position in batch["positions"]],
            dtype=torch.long,
            device=end_logits.device,
        )
        batch_indices = torch.arange(start_logits.size(0), device=start_logits.device)
        feature_scores = (
            start_logits[batch_indices, start_positions]
            + end_logits[batch_indices, end_positions]
        ).float().cpu().tolist()
        for score, row_index, entity_index in zip(
            feature_scores,
            batch["row_indices"],
            batch["entity_indices"],
            strict=True,
        ):
            scores[row_index][entity_index] = max(scores[row_index][entity_index], score)
            feature_count += 1
    torch.cuda.synchronize()
    predictions = []
    exact_values, f1_values = [], []
    for row, entity_scores in zip(rows, scores, strict=True):
        if all(math.isinf(score) for score in entity_scores):
            raise RuntimeError(f"ReCoRD row {row['idx']} has no scored entity")
        best = max(range(len(entity_scores)), key=entity_scores.__getitem__)
        prediction = row["entities"][best]
        answers = list(row["answers"])
        exact = max(record_exact(prediction, answer) for answer in answers)
        f1 = max(record_f1(prediction, answer) for answer in answers)
        exact_values.append(exact)
        f1_values.append(f1)
        predictions.append(
            {
                "id": json_id(row["idx"]),
                "prediction": prediction,
                "gold_answers": answers,
                "exact_match": exact,
                "f1": f1,
            }
        )
    elapsed = time.perf_counter() - started
    return {
        "f1": sum(f1_values) / len(f1_values),
        "exact_match": sum(exact_values) / len(exact_values),
    }, predictions, {
        "rows": len(rows),
        "entity_span_features": feature_count,
        "seconds": elapsed,
        "features_per_second": feature_count / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def primary_score(task: str, metrics: dict[str, float]) -> float:
    if task == "cb":
        return (metrics["accuracy"] + metrics["f1"]) / 2
    if task == "multirc":
        return (metrics["f1_a"] + metrics["exact_match"]) / 2
    if task == "record":
        return (metrics["f1"] + metrics["exact_match"]) / 2
    return metrics["accuracy"]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("multispace", "real"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-proportion", type=float, default=0.06)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    gpu = require_a100()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    predictions_path = args.output_dir / "predictions.jsonl"

    manifest, verified_data = validate_data(
        args.data_root.resolve(), args.data_manifest.resolve(), args.task
    )
    base_model = args.model.resolve()
    weight_path = base_model / "model.safetensors"
    model_sha256 = sha256_file(weight_path)
    if model_sha256 != EXPECTED_MODEL_SHA256[args.variant]:
        raise RuntimeError(
            f"unexpected {args.variant} checkpoint hash: {model_sha256}"
        )
    tokenizer = load_tokenizer(base_model)

    data_task = "wsc.fixed" if args.task == "wsc" else args.task
    train_rows = read_rows(args.data_root / data_task / "train.parquet", args.limit)
    validation_rows = read_rows(
        args.data_root / data_task / "validation.parquet", args.limit
    )
    if not train_rows or not validation_rows:
        raise RuntimeError("SuperGLUE train/validation data cannot be empty")

    model = SuperGlueModel(base_model, args.task)
    model_metadata = validate_model(model, args.variant)
    initialization_sha256 = head_digest(model)
    model.cuda()

    if args.task == "record":
        train_examples = train_rows
        train_collator = RecordTrainCollator(tokenizer, args.max_length)
        validation_features = record_eval_features(validation_rows, args.max_length)
        validation_collator = RecordEvalCollator(tokenizer, args.max_length)
        validation_examples = validation_features
    elif args.task == "copa":
        train_examples = copa_examples(train_rows)
        validation_examples = copa_examples(validation_rows)
        train_collator = CopaCollator(tokenizer, args.max_length)
        validation_collator = train_collator
    else:
        train_examples = sequence_examples(args.task, train_rows)
        validation_examples = sequence_examples(args.task, validation_rows)
        train_collator = SequenceCollator(tokenizer, args.task, args.max_length)
        validation_collator = train_collator

    train_loader = make_loader(
        train_examples, train_collator, args.batch_size, True, args.seed
    )
    validation_loader = make_loader(
        validation_examples,
        validation_collator,
        args.eval_batch_size,
        False,
        args.seed,
    )
    epochs = args.epochs or DEFAULT_EPOCHS[args.task]
    training = train(
        model,
        train_loader,
        epochs,
        args.learning_rate,
        args.weight_decay,
        args.warmup_proportion,
        args.max_train_steps,
    )
    if args.task == "record":
        metrics, predictions, evaluation = evaluate_record(
            model, validation_rows, validation_loader
        )
    else:
        metrics, predictions, evaluation = evaluate_classification(
            model, validation_loader, args.task
        )
    write_jsonl(predictions_path, predictions)

    diagnostics: dict[str, Any] = {}
    if args.task == "rte":
        for diagnostic in ("axb", "axg"):
            rows = read_rows(args.data_root / diagnostic / "test.parquet", args.limit)
            examples = diagnostic_examples(diagnostic, rows)
            loader = make_loader(
                examples,
                SequenceCollator(tokenizer, "rte", args.max_length),
                args.eval_batch_size,
                False,
                args.seed,
            )
            diagnostic_metrics, diagnostic_predictions, diagnostic_runtime = (
                evaluate_classification(model, loader, diagnostic)
            )
            diagnostic_path = args.output_dir / f"{diagnostic}_predictions.jsonl"
            write_jsonl(diagnostic_path, diagnostic_predictions)
            diagnostics[diagnostic] = {
                "metrics": diagnostic_metrics,
                "runtime": diagnostic_runtime,
                "predictions": str(diagnostic_path.resolve()),
                "included_in_primary_aggregate": False,
                "transfer_head": "rte",
            }

    protocol = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_manifest": str(args.data_manifest.resolve()),
        "data_manifest_sha256": sha256_file(args.data_manifest.resolve()),
        "verified_data": verified_data,
        "training_split": "train",
        "evaluation_split": "validation",
        "test_policy": manifest["test_split_policy"],
        "metric_implementation": METRIC_IMPLEMENTATION,
        "sequence_length": args.max_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_proportion": args.warmup_proportion,
        "scheduler": "cosine",
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
        "precision": "FP32",
        "padding": "left",
        "pooling": "final_non_padding_token",
        "head": "BabyLM ClassifierHead; ReCoRD applies it token-wise",
        "record_protocol": (
            "extractive start/end training; validation predictions are constrained "
            "to provided entity spans and scored with official normalized F1/EM"
        ),
        "deterministic_algorithms": True,
        "preflight": args.preflight,
        "limit": args.limit,
    }
    report = {
        "status": "ok",
        "task": args.task,
        "variant": args.variant,
        "seed": args.seed,
        "model": str(base_model),
        "model_safetensors_sha256": model_sha256,
        "model_metadata": model_metadata,
        "task_head_initialization_sha256": initialization_sha256,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training": training,
        "metrics": metrics,
        "primary_score": primary_score(args.task, metrics),
        "diagnostics": diagnostics,
        "evaluation": evaluation,
        "predictions": str(predictions_path.resolve()),
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "gpu": gpu,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
