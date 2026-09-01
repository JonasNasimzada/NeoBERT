#!/usr/bin/env python3
"""Full-model transfer probe for deterministic masked associative recall."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

import masked_mqar as mqar


PROTOCOL_VERSION = "trained-masked-mqar-v1"
DEFAULT_STEPS = 2160
DEFAULT_WARMUP_STEPS = 108
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_MINIMUM_LEARNING_RATE = 5e-6
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_TRAIN_BATCH_TOKENS = 4096
DEFAULT_VALIDATION_EXAMPLES_PER_CELL = 64
DEFAULT_TEST_EXAMPLES_PER_CELL = 256
SPLIT_OFFSETS = {"train": 104_729, "validation": 209_759, "test": 314_777}


def split_seed(transfer_seed: int, split: str) -> int:
    if transfer_seed < 0:
        raise ValueError("transfer seed must be nonnegative")
    if split not in SPLIT_OFFSETS:
        raise ValueError(f"unknown split {split!r}")
    return transfer_seed * 1_000_003 + SPLIT_OFFSETS[split]


def curriculum_indices(
    cell_count: int,
    steps: int,
    *,
    seed: int,
) -> tuple[int, ...]:
    """Deterministically shuffle every complete grid pass independently."""
    if cell_count <= 0 or steps <= 0:
        raise ValueError("cell_count and steps must be positive")
    schedule: list[int] = []
    cycle = 0
    while len(schedule) < steps:
        order = list(range(cell_count))
        material = f"{PROTOCOL_VERSION}|curriculum|{seed}|{cycle}".encode()
        cycle_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        random.Random(cycle_seed).shuffle(order)
        schedule.extend(order)
        cycle += 1
    return tuple(schedule[:steps])


def example_identity(input_ids: Sequence[int], target_token_id: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", int(target_token_id)))
    digest.update(struct.pack(f"<{len(input_ids)}I", *input_ids))
    return digest.digest()


class SplitTracker:
    """Build an ordered split fingerprint and reject generated collisions."""

    def __init__(self, name: str, seed: int, global_identities: dict[bytes, str]):
        self.name = name
        self.seed = seed
        self.digest = hashlib.sha256(
            f"{PROTOCOL_VERSION}|{name}|{seed}".encode("utf-8")
        )
        self.identities: set[bytes] = set()
        self.global_identities = global_identities
        self.examples = 0
        self.token_positions = 0

    def add(self, cell: mqar.CellSpec, example_index: int, example) -> None:
        identity = example_identity(example.input_ids, example.target_token_id)
        previous_split = self.global_identities.get(identity)
        if previous_split is not None:
            raise AssertionError(
                f"generated example collision between {previous_split} and {self.name}"
            )
        if identity in self.identities:
            raise AssertionError(f"duplicate generated example within {self.name}")
        self.identities.add(identity)
        self.global_identities[identity] = self.name
        self.digest.update(self.name.encode("ascii"))
        mqar.update_dataset_fingerprint(self.digest, cell, example_index, example)
        self.examples += 1
        self.token_positions += len(example.input_ids)

    def manifest(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "seed": self.seed,
            "examples": self.examples,
            "token_positions": self.token_positions,
            "dataset_sha256": self.digest.hexdigest(),
            "unique_examples": len(self.identities),
        }


def generate_examples(
    cell: mqar.CellSpec,
    *,
    start_index: int,
    count: int,
    seed: int,
    candidate_token_ids: Sequence[int],
    markers: mqar.MarkerTokens,
    tracker: SplitTracker,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    labels = []
    for example_index in range(start_index, start_index + count):
        example = mqar.generate_example(
            cell,
            example_index,
            seed=seed,
            candidate_token_ids=candidate_token_ids,
            markers=markers,
        )
        tracker.add(cell, example_index, example)
        rows.append(example.input_ids)
        labels.append(example.target_token_id)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def learning_rate_at_step(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if not 0 <= step < total_steps:
        raise ValueError("step lies outside optimizer schedule")
    if warmup_steps > 0 and step < warmup_steps:
        return peak_learning_rate * (step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_learning_rate + (peak_learning_rate - minimum_learning_rate) * cosine


def train_model(
    model,
    grid: Sequence[mqar.CellSpec],
    *,
    steps: int,
    batch_tokens: int,
    transfer_seed: int,
    train_seed: int,
    candidate_token_ids: Sequence[int],
    markers: mqar.MarkerTokens,
    tracker: SplitTracker,
    learning_rate: float,
    minimum_learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    log_every: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=False,
    )
    schedule = curriculum_indices(len(grid), steps, seed=transfer_seed)
    next_example_index: dict[str, int] = defaultdict(int)
    history: list[dict] = []
    interval_loss = 0.0
    interval_correct = 0
    interval_examples = 0
    interval_started = time.perf_counter()
    total_positions = 0
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()

    for step, cell_index in enumerate(schedule):
        cell = grid[cell_index]
        if batch_tokens < cell.context_length or batch_tokens % cell.context_length:
            raise ValueError(
                f"train batch token budget {batch_tokens} is incompatible with "
                f"context {cell.context_length}"
            )
        batch_size = batch_tokens // cell.context_length
        start_index = next_example_index[cell.key]
        input_ids, labels = generate_examples(
            cell,
            start_index=start_index,
            count=batch_size,
            seed=train_seed,
            candidate_token_ids=candidate_token_ids,
            markers=markers,
            tracker=tracker,
        )
        next_example_index[cell.key] += batch_size
        input_ids = input_ids.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        current_lr = learning_rate_at_step(
            step,
            total_steps=steps,
            warmup_steps=warmup_steps,
            peak_learning_rate=learning_rate,
            minimum_learning_rate=minimum_learning_rate,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            selected_logits = model(input_ids)["logits"][:, -2, :].float()
            loss = F.cross_entropy(selected_logits, labels)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite transfer update at step {step + 1}: "
                f"loss={loss.item()}, grad_norm={gradient_norm.item()}"
            )
        optimizer.step()

        batch_correct = int(selected_logits.argmax(dim=-1).eq(labels).sum().item())
        batch_loss = float(loss.item())
        interval_loss += batch_loss * batch_size
        interval_correct += batch_correct
        interval_examples += batch_size
        total_positions += int(input_ids.numel())

        if (step + 1) % log_every == 0 or step + 1 == steps:
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - interval_started
            record = {
                "optimizer_step": step + 1,
                "learning_rate": current_lr,
                "mean_training_loss": interval_loss / interval_examples,
                "training_accuracy": interval_correct / interval_examples,
                "examples": interval_examples,
                "elapsed_seconds": elapsed,
                "examples_per_second": interval_examples / elapsed,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            interval_loss = 0.0
            interval_correct = 0
            interval_examples = 0
            interval_started = time.perf_counter()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - training_started
    return history, {
        "optimizer_steps": steps,
        "training_token_positions": total_positions,
        "elapsed_seconds": elapsed,
        "token_positions_per_second": total_positions / elapsed,
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def evaluate_split(
    model,
    grid: Sequence[mqar.CellSpec],
    *,
    examples_per_cell: int,
    batch_tokens: int,
    seed: int,
    candidate_token_ids: Sequence[int],
    markers: mqar.MarkerTokens,
    tracker: SplitTracker,
    device: torch.device,
) -> dict:
    model.eval()
    reports = []
    for cell in grid:
        if batch_tokens < cell.context_length or batch_tokens % cell.context_length:
            raise ValueError(
                f"evaluation batch token budget is incompatible with context "
                f"{cell.context_length}"
            )
        input_ids, labels = generate_examples(
            cell,
            start_index=0,
            count=examples_per_cell,
            seed=seed,
            candidate_token_ids=candidate_token_ids,
            markers=markers,
            tracker=tracker,
        )
        metrics = mqar.evaluate_cell(
            model,
            input_ids,
            labels,
            batch_size=batch_tokens // cell.context_length,
            device=device,
        )
        reports.append(
            {
                "cell": cell.key,
                "context_length": cell.context_length,
                "binding_count": cell.binding_count,
                "distractor_count": cell.distractor_count,
                "query_distance": cell.query_distance,
                "query_distance_fraction": cell.distance_fraction,
                "difficulty": cell.difficulty,
                **metrics,
            }
        )
    return {
        "cells": {report["cell"]: report for report in reports},
        "summaries": mqar.build_summaries(reports),
        "split_manifest": tracker.manifest(),
    }


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(mqar.MODEL_CONTRACTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transfer-seed", type=int, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=list(mqar.DEFAULT_CONTEXTS))
    parser.add_argument("--bindings", type=int, nargs="+", default=list(mqar.DEFAULT_BINDINGS))
    parser.add_argument(
        "--distractors", type=int, nargs="+", default=list(mqar.DEFAULT_DISTRACTORS)
    )
    parser.add_argument(
        "--distance-fractions",
        type=mqar.parse_fraction,
        nargs="+",
        default=list(mqar.DEFAULT_DISTANCE_FRACTIONS),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--train-batch-tokens", type=int, default=DEFAULT_TRAIN_BATCH_TOKENS)
    parser.add_argument("--eval-batch-tokens", type=int, default=mqar.DEFAULT_BATCH_TOKENS)
    parser.add_argument(
        "--validation-examples-per-cell",
        type=int,
        default=DEFAULT_VALIDATION_EXAMPLES_PER_CELL,
    )
    parser.add_argument(
        "--test-examples-per-cell", type=int, default=DEFAULT_TEST_EXAMPLES_PER_CELL
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--minimum-learning-rate", type=float, default=DEFAULT_MINIMUM_LEARNING_RATE
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--log-every", type=int, default=108)
    parser.add_argument("--no-save-model", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.transfer_seed < 0:
        raise ValueError("transfer-seed must be nonnegative")
    positive_integers = {
        "steps": args.steps,
        "train-batch-tokens": args.train_batch_tokens,
        "eval-batch-tokens": args.eval_batch_tokens,
        "validation-examples-per-cell": args.validation_examples_per_cell,
        "test-examples-per-cell": args.test_examples_per_cell,
        "log-every": args.log_every,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.warmup_steps <= args.steps:
        raise ValueError("warmup-steps must lie in [0, steps]")
    if not 0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError("learning-rate bounds are invalid")
    if args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("weight decay and max gradient norm are invalid")

    grid = mqar.make_grid(
        args.contexts, args.bindings, args.distractors, args.distance_fractions
    )
    device, runtime = mqar.validate_a100_bf16_runtime()
    original_training = mqar.validate_training_completion(args.model)
    from transformers import AutoTokenizer

    from neobert.model import NeoBERTLMHead

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    markers = mqar.resolve_marker_tokens(tokenizer)
    candidate_ids = mqar.build_candidate_token_ids(tokenizer, markers)
    model = NeoBERTLMHead.from_pretrained(str(args.model), local_files_only=True)
    model_contract = mqar.validate_checkpoint_contract(model, tokenizer, args.variant)
    if max(args.contexts) > model.config.max_length:
        raise ValueError("probe context exceeds checkpoint max_length")
    model.to(device)
    torch.manual_seed(args.transfer_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_seed = split_seed(args.transfer_seed, "train")
    validation_seed = split_seed(args.transfer_seed, "validation")
    test_seed = split_seed(args.transfer_seed, "test")
    global_identities: dict[bytes, str] = {}
    train_tracker = SplitTracker("train", train_seed, global_identities)
    validation_tracker = SplitTracker(
        "validation", validation_seed, global_identities
    )
    test_tracker = SplitTracker("test", test_seed, global_identities)
    started = datetime.now(timezone.utc)

    history, training_metrics = train_model(
        model,
        grid,
        steps=args.steps,
        batch_tokens=args.train_batch_tokens,
        transfer_seed=args.transfer_seed,
        train_seed=train_seed,
        candidate_token_ids=candidate_ids,
        markers=markers,
        tracker=train_tracker,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        device=device,
    )
    validation = evaluate_split(
        model,
        grid,
        examples_per_cell=args.validation_examples_per_cell,
        batch_tokens=args.eval_batch_tokens,
        seed=validation_seed,
        candidate_token_ids=candidate_ids,
        markers=markers,
        tracker=validation_tracker,
        device=device,
    )
    test = evaluate_split(
        model,
        grid,
        examples_per_cell=args.test_examples_per_cell,
        batch_tokens=args.eval_batch_tokens,
        seed=test_seed,
        candidate_token_ids=candidate_ids,
        markers=markers,
        tracker=test_tracker,
        device=device,
    )
    split_manifests = {
        "train": train_tracker.manifest(),
        "validation": validation_tracker.manifest(),
        "test": test_tracker.manifest(),
    }
    fingerprints = {
        manifest["dataset_sha256"] for manifest in split_manifests.values()
    }
    if len(fingerprints) != 3:
        raise AssertionError("train/validation/test fingerprints are not disjoint")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_model:
        final_model = output_dir / "final_model"
        model.save_pretrained(final_model, safe_serialization=True)
        tokenizer.save_pretrained(final_model)
    report = {
        "benchmark": "trained-masked-mqar",
        "protocol_version": PROTOCOL_VERSION,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_model_path": str(args.model.resolve()),
        "model": model_contract,
        "original_pretraining_completion": original_training,
        "transfer_completion": {
            "completed_schedule": True,
            "optimizer_steps": args.steps,
        },
        "runtime": runtime,
        "curriculum": {
            "transfer_seed": args.transfer_seed,
            "contexts": list(args.contexts),
            "binding_counts": list(args.bindings),
            "distractor_counts": list(args.distractors),
            "query_distance_fractions": [str(value) for value in args.distance_fractions],
            "grid_cells": len(grid),
            "optimizer_steps": args.steps,
            "complete_grid_passes": args.steps // len(grid),
            "train_batch_tokens": args.train_batch_tokens,
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "validation_examples_per_cell": args.validation_examples_per_cell,
            "test_examples_per_cell": args.test_examples_per_cell,
        },
        "split_manifests": split_manifests,
        "training_metrics": training_metrics,
        "training_history": history,
        "validation": validation,
        "test": test,
    }
    report_path = output_dir / "report.json"
    atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "variant": args.variant,
                "transfer_seed": args.transfer_seed,
                "test_accuracy": test["summaries"]["overall"]["micro_accuracy"],
                "test_masked_nll": test["summaries"]["overall"]["micro_masked_nll"],
                "split_fingerprints": {
                    name: manifest["dataset_sha256"]
                    for name, manifest in split_manifests.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
