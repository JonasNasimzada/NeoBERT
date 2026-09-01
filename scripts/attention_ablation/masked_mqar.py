#!/usr/bin/env python3
"""Evaluate a completed NeoBERT MLM checkpoint on deterministic masked MQAR.

Each example contains randomly paired, single-token key/value bindings followed
by ``query_key [MASK]``.  Keys, values, and explicit distractor tokens are
disjoint within an example.  The input grid and its SHA-256 fingerprint depend
only on the tokenizer and protocol arguments, allowing independently scheduled
models to be checked for exact paired-input identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import struct
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


PROTOCOL_VERSION = "masked-mqar-v1"
EXPECTED_TRAINABLE_PARAMETERS = 99_985_152
EXPECTED_VOCAB_SIZE = 30_522
EXPECTED_HIDDEN_SIZE = 768
EXPECTED_LAYERS = 9
EXPECTED_HEADS = 12
DEFAULT_CONTEXTS = (128, 256, 512, 1024)
DEFAULT_BINDINGS = (4, 8, 16)
DEFAULT_DISTRACTORS = (0, 16, 32)
DEFAULT_DISTANCE_FRACTIONS = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
DEFAULT_EXAMPLES_PER_CELL = 256
DEFAULT_BATCH_TOKENS = 4096

MODEL_CONTRACTS = {
    "multispace-flash": {
        "attention_space": "multispace",
        "attention_backend": "flash",
        "intermediate_size": 2464,
    },
    "real-flash": {
        "attention_space": "real",
        "attention_backend": "flash",
        "intermediate_size": 4000,
    },
}


@dataclass(frozen=True)
class MarkerTokens:
    cls: int
    sep: int
    mask: int
    filler: int
    relation: int
    terminator: int


@dataclass(frozen=True)
class CellSpec:
    context_length: int
    binding_count: int
    distractor_count: int
    distance_numerator: int
    distance_denominator: int
    query_distance: int
    difficulty: str

    @property
    def distance_fraction(self) -> str:
        return f"{self.distance_numerator}/{self.distance_denominator}"

    @property
    def key(self) -> str:
        percent = round(100 * self.distance_numerator / self.distance_denominator)
        return (
            f"c{self.context_length}_b{self.binding_count}_"
            f"d{self.distractor_count}_q{percent}"
        )


@dataclass(frozen=True)
class GeneratedExample:
    input_ids: tuple[int, ...]
    target_token_id: int
    query_key_token_id: int
    mask_position: int
    target_value_position: int
    keys: tuple[int, ...]
    values: tuple[int, ...]
    distractors: tuple[int, ...]


def parse_fraction(value: str) -> Fraction:
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(
            f"distance fraction must look like 1/4; got {value!r}"
        ) from error
    if not 0 < fraction < 1:
        raise argparse.ArgumentTypeError("distance fractions must be in (0, 1)")
    return fraction


def canonical_variant(value: str) -> str:
    variant = value.strip().lower().replace("_", "-")
    if variant not in MODEL_CONTRACTS:
        choices = ", ".join(MODEL_CONTRACTS)
        raise ValueError(f"unknown paired model {value!r}; expected one of {choices}")
    return variant


def make_grid(
    contexts: Sequence[int],
    bindings: Sequence[int],
    distractors: Sequence[int],
    distance_fractions: Sequence[Fraction],
) -> tuple[CellSpec, ...]:
    """Create the complete factorial grid with a documented difficulty band."""
    if not contexts or not bindings or not distractors or not distance_fractions:
        raise ValueError("every MQAR grid dimension must be non-empty")
    if len(set(contexts)) != len(contexts):
        raise ValueError("context lengths must be unique")
    if len(set(bindings)) != len(bindings):
        raise ValueError("binding counts must be unique")
    if len(set(distractors)) != len(distractors):
        raise ValueError("distractor counts must be unique")
    if len(set(distance_fractions)) != len(distance_fractions):
        raise ValueError("query-distance fractions must be unique")
    if any(value <= 0 for value in contexts):
        raise ValueError("context lengths must be positive")
    if any(value <= 0 for value in bindings):
        raise ValueError("binding counts must be positive")
    if any(value < 0 for value in distractors):
        raise ValueError("distractor counts must be nonnegative")

    binding_rank = {value: rank for rank, value in enumerate(sorted(bindings))}
    distractor_rank = {
        value: rank for rank, value in enumerate(sorted(distractors))
    }
    distance_rank = {
        value: rank for rank, value in enumerate(sorted(distance_fractions))
    }
    maximum_rank = (
        max(binding_rank.values())
        + max(distractor_rank.values())
        + max(distance_rank.values())
    )
    cells: list[CellSpec] = []
    for context in contexts:
        for binding_count in bindings:
            for distractor_count in distractors:
                for distance in distance_fractions:
                    query_distance = context * distance.numerator // distance.denominator
                    if query_distance <= 3 or query_distance >= context - 4:
                        raise ValueError(
                            f"context {context} cannot realize query distance {distance}"
                        )
                    rank_sum = (
                        binding_rank[binding_count]
                        + distractor_rank[distractor_count]
                        + distance_rank[distance]
                    )
                    normalized_rank = rank_sum / maximum_rank if maximum_rank else 0.0
                    difficulty = (
                        "easy"
                        if normalized_rank < 1 / 3
                        else "medium" if normalized_rank < 2 / 3 else "hard"
                    )
                    cells.append(
                        CellSpec(
                            context_length=context,
                            binding_count=binding_count,
                            distractor_count=distractor_count,
                            distance_numerator=distance.numerator,
                            distance_denominator=distance.denominator,
                            query_distance=query_distance,
                            difficulty=difficulty,
                        )
                    )
    return tuple(cells)


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer does not provide required marker token {token!r}")
    return int(token_id)


def resolve_marker_tokens(tokenizer) -> MarkerTokens:
    required_specials = {
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "mask_token_id": tokenizer.mask_token_id,
    }
    missing = [name for name, value in required_specials.items() if value is None]
    if missing:
        raise ValueError(f"tokenizer is missing required special IDs: {', '.join(missing)}")
    return MarkerTokens(
        cls=int(tokenizer.cls_token_id),
        sep=int(tokenizer.sep_token_id),
        mask=int(tokenizer.mask_token_id),
        filler=_token_id(tokenizer, "."),
        relation=_token_id(tokenizer, ":"),
        terminator=_token_id(tokenizer, ";"),
    )


def build_candidate_token_ids(
    tokenizer,
    markers: MarkerTokens,
) -> tuple[int, ...]:
    """Select known standalone alphabetic WordPieces as atomic symbols."""
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    marker_ids = set(asdict(markers).values())
    candidates = []
    for token, token_id in tokenizer.get_vocab().items():
        token_id = int(token_id)
        if token_id in special_ids or token_id in marker_ids:
            continue
        if token.startswith("##") or token.startswith("["):
            continue
        if len(token) < 2 or not token.isalpha():
            continue
        candidates.append(token_id)
    candidates = sorted(set(candidates))
    if len(candidates) < 256:
        raise ValueError(
            f"tokenizer yielded only {len(candidates)} suitable atomic symbols"
        )
    return tuple(candidates)


def _example_seed(seed: int, cell: CellSpec, example_index: int) -> int:
    material = (
        f"{PROTOCOL_VERSION}|{seed}|{cell.key}|{cell.query_distance}|{example_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def generate_example(
    cell: CellSpec,
    example_index: int,
    *,
    seed: int,
    candidate_token_ids: Sequence[int],
    markers: MarkerTokens,
) -> GeneratedExample:
    """Generate one deterministic item with exact target-to-query distance."""
    if example_index < 0:
        raise ValueError("example_index must be nonnegative")
    needed_symbols = 2 * cell.binding_count + cell.distractor_count
    if needed_symbols > len(candidate_token_ids):
        raise ValueError(
            f"cell needs {needed_symbols} disjoint symbols, but pool has "
            f"{len(candidate_token_ids)}"
        )
    minimum_positions = 1 + 4 * cell.binding_count + cell.distractor_count + 3
    if minimum_positions > cell.context_length:
        raise ValueError(
            f"{cell.key} needs at least {minimum_positions} positions for its bindings "
            f"and distractors"
        )

    rng = random.Random(_example_seed(seed, cell, example_index))
    symbols = rng.sample(candidate_token_ids, needed_symbols)
    keys = tuple(symbols[: cell.binding_count])
    values = tuple(symbols[cell.binding_count : 2 * cell.binding_count])
    explicit_distractors = tuple(symbols[2 * cell.binding_count :])
    if set(keys) & set(values) or set(keys) & set(explicit_distractors):
        raise AssertionError("generated key/value/distractor pools overlap")
    if set(values) & set(explicit_distractors):
        raise AssertionError("generated value/distractor pools overlap")

    target_index = rng.randrange(cell.binding_count)
    mask_position = cell.context_length - 2
    query_key_position = mask_position - 1
    target_value_position = mask_position - cell.query_distance
    target_start = target_value_position - 2
    target_positions = set(range(target_start, target_start + 4))
    reserved_positions = {0, query_key_position, mask_position, cell.context_length - 1}
    if target_start < 1 or target_start + 3 >= query_key_position:
        raise ValueError(f"{cell.key} places the target binding outside usable positions")
    if target_positions & reserved_positions:
        raise ValueError(f"{cell.key} overlaps target binding and query")

    input_ids = [markers.filler] * cell.context_length
    input_ids[0] = markers.cls
    input_ids[query_key_position] = keys[target_index]
    input_ids[mask_position] = markers.mask
    input_ids[-1] = markers.sep
    occupied = reserved_positions | target_positions

    def place_binding(start: int, key: int, value: int) -> None:
        input_ids[start : start + 4] = [
            key,
            markers.relation,
            value,
            markers.terminator,
        ]

    place_binding(target_start, keys[target_index], values[target_index])

    # Four-token slots make feasibility deterministic even when the controlled
    # target block is not aligned to the slot lattice.  Randomizing their order
    # prevents pair number from becoming a location cue.
    candidate_starts = list(range(1, query_key_position - 3, 4))
    rng.shuffle(candidate_starts)
    other_indices = [index for index in range(cell.binding_count) if index != target_index]
    rng.shuffle(other_indices)
    for binding_index in other_indices:
        for start in candidate_starts:
            positions = set(range(start, start + 4))
            if positions.isdisjoint(occupied):
                place_binding(start, keys[binding_index], values[binding_index])
                occupied.update(positions)
                break
        else:
            raise ValueError(f"{cell.key} cannot place all non-overlapping bindings")

    available_positions = [
        position
        for position in range(1, query_key_position)
        if position not in occupied
    ]
    rng.shuffle(available_positions)
    if len(available_positions) < cell.distractor_count:
        raise ValueError(f"{cell.key} cannot place all explicit distractors")
    for position, token_id in zip(available_positions, explicit_distractors):
        input_ids[position] = token_id

    if input_ids.count(markers.mask) != 1:
        raise AssertionError("an MQAR item must contain exactly one mask")
    if mask_position - target_value_position != cell.query_distance:
        raise AssertionError("target-to-mask distance does not match the cell")
    return GeneratedExample(
        input_ids=tuple(input_ids),
        target_token_id=values[target_index],
        query_key_token_id=keys[target_index],
        mask_position=mask_position,
        target_value_position=target_value_position,
        keys=keys,
        values=values,
        distractors=explicit_distractors,
    )


def update_dataset_fingerprint(
    digest,
    cell: CellSpec,
    example_index: int,
    example: GeneratedExample,
) -> None:
    header = (
        cell.context_length,
        cell.binding_count,
        cell.distractor_count,
        cell.query_distance,
        example_index,
        example.target_token_id,
    )
    digest.update(struct.pack("<6I", *header))
    digest.update(struct.pack(f"<{len(example.input_ids)}I", *example.input_ids))


def generate_cell_tensor(
    cell: CellSpec,
    *,
    examples_per_cell: int,
    seed: int,
    candidate_token_ids: Sequence[int],
    markers: MarkerTokens,
    fingerprint=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if examples_per_cell <= 0:
        raise ValueError("examples_per_cell must be positive")
    rows: list[tuple[int, ...]] = []
    labels: list[int] = []
    for example_index in range(examples_per_cell):
        example = generate_example(
            cell,
            example_index,
            seed=seed,
            candidate_token_ids=candidate_token_ids,
            markers=markers,
        )
        rows.append(example.input_ids)
        labels.append(example.target_token_id)
        if fingerprint is not None:
            update_dataset_fingerprint(fingerprint, cell, example_index, example)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def scalar_parameter_count(module) -> int:
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def validate_training_completion(model_path: Path) -> Mapping:
    summary_path = model_path / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"completed-training summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved_config = summary.get("resolved_config", {})
    trainer = resolved_config.get("trainer", {}) if isinstance(resolved_config, dict) else {}
    completed = summary.get("completed_schedule") is True
    if not completed and "train/completed_schedule" in summary:
        completed = summary["train/completed_schedule"] is True
    if not completed:
        try:
            completed = int(summary["optimizer_steps"]) == int(trainer["max_steps"])
        except (KeyError, TypeError, ValueError):
            completed = False
    if not completed:
        raise RuntimeError(f"checkpoint did not complete its training schedule: {summary_path}")
    return {
        "path": str(summary_path.resolve()),
        "optimizer_steps": summary.get("optimizer_steps"),
        "completed_schedule": True,
    }


def validate_checkpoint_contract(model, tokenizer, variant: str) -> dict:
    contract = MODEL_CONTRACTS[canonical_variant(variant)]
    config = model.config
    expected_scalars = {
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "num_hidden_layers": EXPECTED_LAYERS,
        "num_attention_heads": EXPECTED_HEADS,
        "intermediate_size": contract["intermediate_size"],
        "tie_word_embeddings": True,
        "lm_head_bias": False,
    }
    for name, expected in expected_scalars.items():
        actual = getattr(config, name, None)
        if actual != expected:
            raise AssertionError(f"checkpoint {name}={actual!r}; expected {expected!r}")
    expected_spaces = [contract["attention_space"]] * EXPECTED_LAYERS
    expected_backends = [contract["attention_backend"]] * EXPECTED_LAYERS
    if list(config.attention_spaces) != expected_spaces:
        raise AssertionError(
            f"checkpoint attention spaces are {config.attention_spaces!r}; "
            f"expected {expected_spaces!r}"
        )
    if list(config.attention_backends) != expected_backends:
        raise AssertionError(
            f"checkpoint attention backends are {config.attention_backends!r}; "
            f"expected {expected_backends!r}"
        )
    parameter_count = scalar_parameter_count(model)
    if parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(
            f"checkpoint has {parameter_count:,} trainable real scalars; expected "
            f"{EXPECTED_TRAINABLE_PARAMETERS:,}"
        )
    if model.decoder.weight is not model.model.encoder.weight:
        raise AssertionError("checkpoint input/output embeddings are not tied")
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise AssertionError(
            f"tokenizer has {len(tokenizer):,} entries; expected {EXPECTED_VOCAB_SIZE:,}"
        )
    parameter_dtypes = {
        parameter.real.dtype if parameter.is_complex() else parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point() or parameter.is_complex()
    }
    if parameter_dtypes != {torch.float32}:
        raise AssertionError(
            "checkpoint parameters must load in FP32 before BF16 autocast; got "
            + ", ".join(sorted(str(dtype) for dtype in parameter_dtypes))
        )
    return {
        "variant": variant,
        "trainable_parameters": parameter_count,
        "attention_space": contract["attention_space"],
        "attention_backend": contract["attention_backend"],
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "heads": config.num_attention_heads,
        "head_dimension": config.hidden_size // config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_length": config.max_length,
        "tied_embeddings": True,
    }


def validate_a100_bf16_runtime() -> tuple[torch.device, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("masked MQAR requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("masked MQAR requires CUDA BF16 support")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    if "A100" not in properties.name.upper():
        raise RuntimeError(f"masked MQAR requires an A100; found {properties.name}")
    return device, {
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device_total_memory_bytes": int(properties.total_memory),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "autocast_dtype": "torch.bfloat16",
        "parameter_dtype": "torch.float32",
    }


def _forward_mask_logits(model, input_ids: torch.Tensor, mask_position: int):
    logits = model(input_ids)["logits"]
    return logits[:, mask_position, :].float()


def warm_up(
    model,
    input_ids: torch.Tensor,
    *,
    batch_size: int,
    iterations: int,
    device: torch.device,
) -> None:
    warmup_ids = input_ids[:batch_size].to(device=device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for _ in range(iterations):
            _forward_mask_logits(model, warmup_ids, input_ids.shape[1] - 2)
    torch.cuda.synchronize(device)


def evaluate_cell(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, int | float]:
    if input_ids.shape[0] != labels.shape[0]:
        raise ValueError("input and label counts differ")
    correct = torch.zeros((), dtype=torch.int64, device=device)
    nll_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct_chunks: list[torch.Tensor] = []
    nll_chunks: list[torch.Tensor] = []
    batches = 0
    mask_position = input_ids.shape[1] - 2
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for start in range(0, input_ids.shape[0], batch_size):
            stop = min(start + batch_size, input_ids.shape[0])
            batch_ids = input_ids[start:stop].to(device=device, non_blocking=True)
            batch_labels = labels[start:stop].to(device=device, non_blocking=True)
            selected_logits = _forward_mask_logits(model, batch_ids, mask_position)
            batch_nll = F.cross_entropy(
                selected_logits, batch_labels, reduction="none"
            )
            batch_correct = selected_logits.argmax(dim=-1).eq(batch_labels)
            nll_sum += batch_nll.sum().double()
            correct += batch_correct.sum()
            nll_chunks.append(batch_nll)
            correct_chunks.append(batch_correct)
            batches += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    example_count = int(labels.numel())
    token_positions = int(input_ids.numel())
    nll_value = float(nll_sum.item())
    correct_value = int(correct.item())
    example_nll = [
        float(value) for value in torch.cat(nll_chunks).cpu().tolist()
    ]
    example_correct = [
        bool(value) for value in torch.cat(correct_chunks).cpu().tolist()
    ]
    return {
        "examples": example_count,
        "correct": correct_value,
        "nll_sum": nll_value,
        "accuracy": correct_value / example_count,
        "masked_nll": nll_value / example_count,
        "masked_pseudo_perplexity": math.exp(min(nll_value / example_count, 80.0)),
        "batches": batches,
        "batch_size": batch_size,
        "evaluated_token_positions": token_positions,
        "elapsed_seconds": elapsed,
        "examples_per_second": example_count / elapsed,
        "token_positions_per_second": token_positions / elapsed,
        "mean_batch_latency_ms": 1000.0 * elapsed / batches,
        "mean_example_latency_ms": 1000.0 * elapsed / example_count,
        # Retaining paired item outcomes permits confidence intervals on the
        # multispace-minus-real difference without rerunning either A100 job.
        "example_correct": example_correct,
        "example_nll": example_nll,
    }


def summarize_cells(cells: Sequence[Mapping]) -> dict[str, int | float]:
    if not cells:
        raise ValueError("cannot summarize an empty MQAR cell collection")
    examples = sum(int(cell["examples"]) for cell in cells)
    correct = sum(int(cell["correct"]) for cell in cells)
    nll_sum = sum(float(cell["nll_sum"]) for cell in cells)
    elapsed = sum(float(cell["elapsed_seconds"]) for cell in cells)
    positions = sum(int(cell["evaluated_token_positions"]) for cell in cells)
    return {
        "cell_count": len(cells),
        "examples": examples,
        "correct": correct,
        "micro_accuracy": correct / examples,
        "micro_masked_nll": nll_sum / examples,
        "macro_cell_accuracy": sum(float(cell["accuracy"]) for cell in cells)
        / len(cells),
        "macro_cell_masked_nll": sum(float(cell["masked_nll"]) for cell in cells)
        / len(cells),
        "elapsed_seconds": elapsed,
        "evaluated_token_positions": positions,
        "examples_per_second": examples / elapsed,
        "token_positions_per_second": positions / elapsed,
    }


def build_summaries(cell_reports: Sequence[Mapping]) -> dict[str, Mapping]:
    dimensions = {
        "by_context_length": "context_length",
        "by_binding_count": "binding_count",
        "by_distractor_count": "distractor_count",
        "by_query_distance_fraction": "query_distance_fraction",
        "by_difficulty": "difficulty",
    }
    summaries: dict[str, Mapping] = {"overall": summarize_cells(cell_reports)}
    for summary_name, field in dimensions.items():
        values = sorted({str(cell[field]) for cell in cell_reports})
        summaries[summary_name] = {
            value: summarize_cells(
                [cell for cell in cell_reports if str(cell[field]) == value]
            )
            for value in values
        }
    return summaries


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
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
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def protocol_manifest(
    *,
    cells: Sequence[CellSpec],
    examples_per_cell: int,
    batch_tokens: int,
    seed: int,
    candidate_ids: Sequence[int],
    markers: MarkerTokens,
    dataset_sha256: str,
) -> dict:
    candidate_digest = hashlib.sha256(
        struct.pack(f"<{len(candidate_ids)}I", *candidate_ids)
    ).hexdigest()
    return {
        "name": "masked multi-query associative recall",
        "version": PROTOCOL_VERSION,
        "format": "[CLS] key : value ; ... query_key [MASK] [SEP]",
        "single_token_keys_and_values": True,
        "within_example_symbol_sets_disjoint": True,
        "random_pairings_per_example": True,
        "seed": seed,
        "contexts": sorted({cell.context_length for cell in cells}),
        "binding_counts": sorted({cell.binding_count for cell in cells}),
        "distractor_counts": sorted({cell.distractor_count for cell in cells}),
        "query_distance_fractions": sorted(
            {cell.distance_fraction for cell in cells}
        ),
        "cell_count": len(cells),
        "examples_per_cell": examples_per_cell,
        "total_examples": len(cells) * examples_per_cell,
        "batch_tokens": batch_tokens,
        "markers": asdict(markers),
        "candidate_pool_size": len(candidate_ids),
        "candidate_pool_sha256": candidate_digest,
        "dataset_sha256": dataset_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=tuple(MODEL_CONTRACTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--bindings", type=int, nargs="+", default=list(DEFAULT_BINDINGS))
    parser.add_argument(
        "--distractors", type=int, nargs="+", default=list(DEFAULT_DISTRACTORS)
    )
    parser.add_argument(
        "--distance-fractions",
        type=parse_fraction,
        nargs="+",
        default=list(DEFAULT_DISTANCE_FRACTIONS),
    )
    parser.add_argument("--examples-per-cell", type=int, default=DEFAULT_EXAMPLES_PER_CELL)
    parser.add_argument("--batch-tokens", type=int, default=DEFAULT_BATCH_TOKENS)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.variant = canonical_variant(args.variant)
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.examples_per_cell <= 0:
        raise ValueError("examples-per-cell must be positive")
    if args.batch_tokens <= 0:
        raise ValueError("batch-tokens must be positive")
    if args.warmup_iterations <= 0:
        raise ValueError("warmup-iterations must be positive")

    cells = make_grid(
        args.contexts,
        args.bindings,
        args.distractors,
        args.distance_fractions,
    )
    for context in args.contexts:
        if args.batch_tokens < context or args.batch_tokens % context:
            raise ValueError(
                f"batch-tokens must be at least and divisible by context {context}"
            )

    device, runtime = validate_a100_bf16_runtime()
    training_completion = validate_training_completion(args.model)

    # Imported only after runtime validation so an accidental CPU launch fails
    # before allocating either 100M-parameter model.
    from transformers import AutoTokenizer

    from neobert.model import NeoBERTLMHead

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    markers = resolve_marker_tokens(tokenizer)
    candidate_ids = build_candidate_token_ids(tokenizer, markers)
    model = NeoBERTLMHead.from_pretrained(str(args.model), local_files_only=True)
    model_contract = validate_checkpoint_contract(model, tokenizer, args.variant)
    if max(args.contexts) > model.config.max_length:
        raise ValueError(
            f"requested context {max(args.contexts)} exceeds model max_length "
            f"{model.config.max_length}"
        )

    model.to(device)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    benchmark_started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    dataset_fingerprint = hashlib.sha256()
    dataset_fingerprint.update(PROTOCOL_VERSION.encode("utf-8"))
    dataset_fingerprint.update(str(args.seed).encode("ascii"))
    cell_reports: list[dict] = []
    memory_by_context: dict[str, dict] = {}

    for context in args.contexts:
        context_cells = [cell for cell in cells if cell.context_length == context]
        batch_size = args.batch_tokens // context
        warmup_inputs, _ = generate_cell_tensor(
            context_cells[0],
            examples_per_cell=max(1, min(batch_size, args.examples_per_cell)),
            seed=args.seed,
            candidate_token_ids=candidate_ids,
            markers=markers,
        )
        warm_up(
            model,
            warmup_inputs,
            batch_size=min(batch_size, warmup_inputs.shape[0]),
            iterations=args.warmup_iterations,
            device=device,
        )
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)

        for cell in context_cells:
            input_ids, labels = generate_cell_tensor(
                cell,
                examples_per_cell=args.examples_per_cell,
                seed=args.seed,
                candidate_token_ids=candidate_ids,
                markers=markers,
                fingerprint=dataset_fingerprint,
            )
            metrics = evaluate_cell(
                model,
                input_ids,
                labels,
                batch_size=batch_size,
                device=device,
            )
            cell_reports.append(
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

        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        memory_by_context[str(context)] = {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": max(
                peak_allocated - baseline_allocated, 0
            ),
            "incremental_peak_reserved_bytes": max(
                peak_reserved - baseline_reserved, 0
            ),
        }

    torch.cuda.synchronize(device)
    benchmark_finished = datetime.now(timezone.utc)
    dataset_sha256 = dataset_fingerprint.hexdigest()
    report = {
        "benchmark": "masked-mqar",
        "benchmark_started_utc": benchmark_started.isoformat(),
        "benchmark_finished_utc": benchmark_finished.isoformat(),
        "wall_elapsed_seconds": time.perf_counter() - wall_started,
        "model_path": str(args.model.resolve()),
        "model": model_contract,
        "training_completion": training_completion,
        "protocol": protocol_manifest(
            cells=cells,
            examples_per_cell=args.examples_per_cell,
            batch_tokens=args.batch_tokens,
            seed=args.seed,
            candidate_ids=candidate_ids,
            markers=markers,
            dataset_sha256=dataset_sha256,
        ),
        "runtime": {
            **runtime,
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "pid": os.getpid(),
        },
        "memory_by_context": memory_by_context,
        "cells": {cell["cell"]: cell for cell in cell_reports},
        "summaries": build_summaries(cell_reports),
    }
    atomic_write_json(args.output, report)
    overall = report["summaries"]["overall"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "variant": args.variant,
                "dataset_sha256": dataset_sha256,
                "examples": overall["examples"],
                "accuracy": overall["micro_accuracy"],
                "masked_nll": overall["micro_masked_nll"],
                "token_positions_per_second": overall["token_positions_per_second"],
                "max_peak_allocated_bytes": max(
                    metrics["peak_allocated_bytes"]
                    for metrics in memory_by_context.values()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
