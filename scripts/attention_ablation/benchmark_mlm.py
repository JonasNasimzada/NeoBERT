#!/usr/bin/env python3
"""Benchmark an exported attention-ablation model on held-out MLM data."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch
from torch.nn import functional as F


EXPECTED_TRAINABLE_PARAMETERS = 17_260_288
DEFAULT_CONTEXT_LENGTHS = (128, 256, 512)
DEFAULT_TOKEN_BUDGET = 1_048_576
DEFAULT_BATCH_TOKENS = 4_096
VARIANT_MATRIX = {
    "complex-native": ("complex", "native"),
    "complex-torch": ("complex", "torch"),
    "complex-flash": ("complex", "flash"),
    "split-native": ("split", "native"),
    "split-torch": ("split", "torch"),
    "dual-native": ("dual", "native"),
    "dual-torch": ("dual", "torch"),
}


def canonical_variant(value: str) -> str:
    variant = value.strip().lower().replace("_", "-")
    if variant not in VARIANT_MATRIX:
        choices = ", ".join(VARIANT_MATRIX)
        raise ValueError(f"unknown variant {value!r}; expected one of: {choices}")
    return variant


def validate_training_completion(
    model_directory: Path,
    *,
    allow_incomplete: bool = False,
) -> dict:
    """Require proof that a final_model export reached its configured schedule."""
    summary_path = model_directory / "training_summary.json"
    if not summary_path.is_file():
        if allow_incomplete:
            return {"status": "allowed_missing", "path": str(summary_path.resolve())}
        raise FileNotFoundError(
            f"training summary is missing: {summary_path}. "
            "Pass --allow-incomplete only for an intentional partial benchmark."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid training summary {summary_path}: {error}") from error
    if not isinstance(summary, dict):
        raise TypeError(f"training summary must contain a JSON object: {summary_path}")

    completion_present = "completed_schedule" in summary
    completed_schedule = summary.get("completed_schedule")
    if not completion_present and "train/completed_schedule" in summary:
        completion_present = True
        completed_schedule = summary["train/completed_schedule"]

    optimizer_steps = summary.get("optimizer_steps")
    resolved_config = summary.get("resolved_config")
    configured_max_steps = None
    if isinstance(resolved_config, dict):
        trainer_config = resolved_config.get("trainer")
        if isinstance(trainer_config, dict):
            configured_max_steps = trainer_config.get("max_steps")

    completed = completed_schedule is True
    if not completion_present:
        try:
            completed = int(optimizer_steps) == int(configured_max_steps)
        except (TypeError, ValueError):
            completed = False

    if not completed and not allow_incomplete:
        raise RuntimeError(
            f"checkpoint did not complete its configured training schedule: {summary_path} "
            f"(completed_schedule={completed_schedule!r}, "
            f"optimizer_steps={optimizer_steps!r}, max_steps={configured_max_steps!r}). "
            "Resume training, or pass --allow-incomplete for an intentional partial benchmark."
        )
    return {
        "status": "complete" if completed else "allowed_incomplete",
        "path": str(summary_path.resolve()),
        "completed_schedule": completed_schedule,
        "optimizer_steps": optimizer_steps,
        "configured_max_steps": configured_max_steps,
    }


def deterministic_mask_inputs(
    input_ids: torch.Tensor,
    *,
    mask_token_id: int,
    special_token_ids: Iterable[int],
    probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace a reproducible Bernoulli sample of non-special tokens by MASK."""
    if input_ids.device.type != "cpu":
        raise ValueError("deterministic masking expects CPU input_ids")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must use an integer dtype")
    if not 0.0 < probability <= 1.0:
        raise ValueError("mask probability must be in (0, 1]")

    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in sorted({int(value) for value in special_token_ids}):
        eligible.logical_and_(input_ids.ne(token_id))

    draws = torch.rand(input_ids.shape, generator=generator)
    masked_positions = eligible & draws.lt(probability)
    corrupted = input_ids.clone()
    corrupted[masked_positions] = int(mask_token_id)
    labels = torch.full_like(input_ids, -100)
    labels[masked_positions] = input_ids[masked_positions]
    return corrupted, labels, eligible


def iter_fixed_token_batches(
    dataset,
    *,
    context_length: int,
    token_budget: int,
    batch_tokens: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """Stream the same flattened token prefix at any requested context length."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if token_budget <= 0 or token_budget % context_length:
        raise ValueError("token_budget must be positive and divisible by context_length")
    if batch_tokens < context_length or batch_tokens % context_length:
        raise ValueError("batch_tokens must be divisible by context_length")

    columns = getattr(dataset, "column_names", None)
    if columns is not None and "input_ids" not in columns:
        raise ValueError("validation split must contain an input_ids column")
    has_document_ids = columns is not None and "document_ids" in columns
    sequence_columns = ("input_ids", "document_ids") if has_document_ids else ("input_ids",)
    buffers: dict[str, list[int]] = {column: [] for column in sequence_columns}
    pending: dict[str, list[list[int]]] = {column: [] for column in sequence_columns}
    batch_size = batch_tokens // context_length
    emitted_tokens = 0

    for row in dataset:
        row_length = len(row["input_ids"])
        for column in sequence_columns:
            values = row[column]
            if len(values) != row_length:
                raise ValueError(f"{column} length does not match input_ids")
            buffers[column].extend(int(value) for value in values)

        while len(buffers["input_ids"]) >= context_length and emitted_tokens < token_budget:
            for column in sequence_columns:
                pending[column].append(buffers[column][:context_length])
                del buffers[column][:context_length]
            emitted_tokens += context_length

            if len(pending["input_ids"]) == batch_size or emitted_tokens == token_budget:
                batch = {
                    "input_ids": torch.tensor(pending["input_ids"], dtype=torch.long)
                }
                if has_document_ids:
                    batch["document_ids"] = torch.tensor(
                        pending["document_ids"],
                        dtype=torch.int32,
                    )
                yield batch
                pending = {column: [] for column in sequence_columns}

            if emitted_tokens == token_budget:
                return

    raise ValueError(
        f"validation split contains fewer than {token_budget:,} usable token positions"
    )


def _masked_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    tokenizer,
    probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokenizer.mask_token_id is None:
        raise ValueError("the checkpoint tokenizer does not define a mask token")
    return deterministic_mask_inputs(
        batch["input_ids"],
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=tokenizer.all_special_ids,
        probability=probability,
        generator=generator,
    )


def _forward(model, input_ids: torch.Tensor, document_ids: torch.Tensor | None):
    if document_ids is None:
        return model(input_ids)["logits"]
    return model(input_ids, document_ids=document_ids)["logits"]


def _warm_up(
    model,
    dataset,
    tokenizer,
    *,
    context_length: int,
    token_budget: int,
    batch_tokens: int,
    probability: float,
    seed: int,
    device: torch.device,
) -> None:
    warmup_budget = min(token_budget, batch_tokens)
    warmup_budget -= warmup_budget % context_length
    batch = next(
        iter_fixed_token_batches(
            dataset,
            context_length=context_length,
            token_budget=warmup_budget,
            batch_tokens=batch_tokens,
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs, _, _ = _masked_batch(
        batch,
        tokenizer=tokenizer,
        probability=probability,
        generator=generator,
    )
    inputs = inputs.to(device=device, non_blocking=True)
    document_ids = batch.get("document_ids")
    if document_ids is not None:
        document_ids = document_ids.to(device=device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        _forward(model, inputs, document_ids)
    torch.cuda.synchronize(device)


def evaluate_context(
    model,
    dataset,
    tokenizer,
    *,
    context_length: int,
    token_budget: int,
    batch_tokens: int,
    probability: float,
    seed: int,
    device: torch.device,
) -> dict[str, int | float]:
    """Evaluate one context length, including an unmeasured warm-up batch."""
    _warm_up(
        model,
        dataset,
        tokenizer,
        context_length=context_length,
        token_budget=token_budget,
        batch_tokens=batch_tokens,
        probability=probability,
        seed=seed + 10_000 + context_length,
        device=device,
    )

    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct_sum = torch.zeros((), dtype=torch.int64, device=device)
    masked_tokens = 0
    eligible_tokens = 0
    evaluated_tokens = 0

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        for batch in iter_fixed_token_batches(
            dataset,
            context_length=context_length,
            token_budget=token_budget,
            batch_tokens=batch_tokens,
        ):
            inputs, labels, eligible = _masked_batch(
                batch,
                tokenizer=tokenizer,
                probability=probability,
                generator=generator,
            )
            selected = labels.ne(-100)
            batch_masked_tokens = int(selected.sum().item())
            masked_tokens += batch_masked_tokens
            eligible_tokens += int(eligible.sum().item())
            evaluated_tokens += inputs.numel()

            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            selected = selected.to(device=device, non_blocking=True)
            document_ids = batch.get("document_ids")
            if document_ids is not None:
                document_ids = document_ids.to(device=device, non_blocking=True)

            logits = _forward(model, inputs, document_ids)
            if batch_masked_tokens:
                selected_logits = logits[selected].float()
                selected_labels = labels[selected]
                loss_sum += F.cross_entropy(
                    selected_logits,
                    selected_labels,
                    reduction="sum",
                ).double()
                correct_sum += selected_logits.argmax(dim=-1).eq(selected_labels).sum()

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    if evaluated_tokens != token_budget:
        raise AssertionError(
            f"evaluated {evaluated_tokens:,} tokens, expected {token_budget:,}"
        )
    if masked_tokens == 0:
        raise RuntimeError("deterministic corruption selected no validation tokens")

    mean_loss = loss_sum.item() / masked_tokens
    accuracy = correct_sum.item() / masked_tokens
    perplexity = math.exp(mean_loss)
    return {
        "context_length": context_length,
        "batch_size": batch_tokens // context_length,
        "evaluated_tokens": evaluated_tokens,
        "eligible_tokens": eligible_tokens,
        "masked_tokens": masked_tokens,
        "masked_fraction": masked_tokens / eligible_tokens,
        "loss": mean_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": evaluated_tokens / elapsed_seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _atomic_write_json(path: Path, payload: Mapping) -> None:
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


def _wandb_identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return identifier[:120] or "attention-ablation"


def _log_to_wandb(report: Mapping, output_path: Path, args) -> None:
    if args.mode == "disabled":
        return
    import wandb

    run = wandb.init(
        id=_wandb_identifier(f"{args.variant}-seed-{args.seed}-heldout-mlm"),
        resume="allow",
        project=args.project,
        entity=args.entity or None,
        name=args.name or f"{args.variant}-heldout-mlm",
        group=args.group or args.variant,
        job_type="benchmark",
        mode=args.mode,
        tags=["benchmark", "heldout-mlm", args.variant],
        config={
            "variant": args.variant,
            "model": str(args.model.resolve()),
            "dataset": str(args.dataset.resolve()),
            "split": args.split,
            "seed": args.seed,
            "contexts": list(args.contexts),
            "token_budget": args.token_budget,
            "batch_tokens": args.batch_tokens,
            "mask_probability": args.mask_probability,
            "allow_incomplete": args.allow_incomplete,
            "training_completion": report["training_completion"]["status"],
        },
    )
    try:
        scalars: dict[str, int | float] = {
            "benchmark/mlm/trainable_parameters": report["trainable_parameters"]
        }
        for context, metrics in report["results"].items():
            prefix = f"benchmark/mlm/context_{context}"
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    scalars[f"{prefix}/{name}"] = value
        run.log(scalars)

        artifact = wandb.Artifact(
            name=_wandb_identifier(
                f"{args.variant}-heldout-mlm-{run.id}"
            ),
            type="benchmark-results",
            metadata={"variant": args.variant, "benchmark": "heldout-mlm"},
        )
        artifact.add_file(str(output_path.resolve()), name="heldout_mlm.json")
        run.log_artifact(artifact)
    finally:
        run.finish()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Exported final_model directory")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared DatasetDict directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report")
    parser.add_argument("--variant", required=True, help="One of the seven attention variants")
    parser.add_argument("--split", default="validation", help="Held-out DatasetDict split")
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
    )
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--batch-tokens", type=int, default=DEFAULT_BATCH_TOKENS)
    parser.add_argument("--mask-probability", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Explicitly permit a missing or partial training summary",
    )
    parser.add_argument("--project", default="complex-attention-ablation")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--group", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--mode",
        "--wandb-mode",
        dest="mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.variant = canonical_variant(args.variant)
    args.contexts = tuple(args.contexts)
    if len(set(args.contexts)) != len(args.contexts):
        raise ValueError("context lengths must be unique")
    for context_length in args.contexts:
        if args.token_budget % context_length:
            raise ValueError("token budget must be divisible by every context length")
        if args.batch_tokens % context_length:
            raise ValueError("batch token budget must be divisible by every context length")
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_mlm.py requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("benchmark_mlm.py requires CUDA BF16 support")

    training_completion = validate_training_completion(
        args.model,
        allow_incomplete=args.allow_incomplete,
    )

    from datasets import DatasetDict, load_from_disk
    from transformers import AutoTokenizer

    from neobert.model import NeoBERTLMHead

    prepared_dataset = load_from_disk(str(args.dataset))
    if not isinstance(prepared_dataset, DatasetDict):
        raise TypeError("--dataset must point to a saved DatasetDict")
    if args.split not in prepared_dataset:
        available = ", ".join(prepared_dataset)
        raise KeyError(f"held-out split {args.split!r} is missing; available: {available}")
    validation_dataset = prepared_dataset[args.split]
    if "input_ids" not in validation_dataset.column_names:
        raise ValueError("held-out split does not contain input_ids")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    if tokenizer.mask_token_id is None:
        raise ValueError("exported tokenizer does not define mask_token_id")

    device = torch.device("cuda")
    # Match training numerics: retain FP32 master parameters and use BF16 only
    # through CUDA autocast during the forward pass.
    model = NeoBERTLMHead.from_pretrained(str(args.model))
    parameter_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if parameter_dtypes != {torch.float32}:
        dtype_names = ", ".join(sorted(str(dtype) for dtype in parameter_dtypes))
        raise AssertionError(
            f"checkpoint parameters must load in FP32 before BF16 autocast; got {dtype_names}"
        )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(
            f"checkpoint has {trainable_parameters:,} trainable parameters; "
            f"expected {EXPECTED_TRAINABLE_PARAMETERS:,}"
        )
    if model.decoder.weight is not model.model.encoder.weight:
        raise AssertionError("checkpoint input and output embeddings are not tied")

    expected_space, expected_backend = VARIANT_MATRIX[args.variant]
    if set(model.config.attention_spaces) != {expected_space}:
        raise AssertionError(
            f"{args.variant} checkpoint does not use only {expected_space} attention"
        )
    if set(model.config.attention_backends) != {expected_backend}:
        raise AssertionError(
            f"{args.variant} checkpoint does not use only {expected_backend} backend"
        )
    if max(args.contexts) > model.config.max_length:
        raise ValueError("requested context exceeds checkpoint max_length")
    if len(tokenizer) != model.config.vocab_size:
        raise AssertionError("tokenizer and model vocabulary sizes differ")

    model.to(device)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results = {}
    for context_length in args.contexts:
        metrics = evaluate_context(
            model,
            validation_dataset,
            tokenizer,
            context_length=context_length,
            token_budget=args.token_budget,
            batch_tokens=args.batch_tokens,
            probability=args.mask_probability,
            seed=args.seed,
            device=device,
        )
        results[str(context_length)] = metrics
        print(
            f"context={context_length}: loss={metrics['loss']:.6f}, "
            f"ppl={metrics['perplexity']:.3f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"tokens/s={metrics['tokens_per_second']:,.1f}, "
            f"peak={metrics['peak_memory_bytes'] / 2**30:.2f} GiB"
        )

    report = {
        "schema_version": 1,
        "benchmark": "heldout_mlm",
        "variant": args.variant,
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "parameter_dtype": "float32",
        "compute_dtype": "bfloat16",
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "mask_probability": args.mask_probability,
        "token_budget_per_context": args.token_budget,
        "batch_tokens": args.batch_tokens,
        "trainable_parameters": trainable_parameters,
        "training_completion": training_completion,
        "results": results,
    }
    _atomic_write_json(args.output, report)
    _log_to_wandb(report, args.output, args)
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
