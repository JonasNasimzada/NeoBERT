#!/usr/bin/env python3
"""GPU preflight for the paired FineWeb-Edu BabyLM 2026 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoProcessor, PreTrainedTokenizerFast


EXPECTED_PARAMETERS = 99_985_152


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected-space", required=True, choices=("real", "multispace"))
    return parser.parse_args()


def load_tokenizer(model_path: Path):
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        return processor.tokenizer if hasattr(processor, "tokenizer") else processor
    except (KeyError, OSError, ValueError):
        return PreTrainedTokenizerFast.from_pretrained(
            model_path,
            padding_side="right",
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BabyLM preflight must run on a CUDA GPU")
    device_name = torch.cuda.get_device_name()
    if "A100" not in device_name.upper():
        raise RuntimeError(f"BabyLM preflight requires an A100; found {device_name}")

    required = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required if not (args.model / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete model export {args.model}: missing {missing}")

    tokenizer = load_tokenizer(args.model)
    batch = tokenizer(
        ["The cat sleeps.", "A dog runs quickly through the park."],
        padding=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    if bool(batch["attention_mask"].all()):
        raise RuntimeError("preflight did not construct a padded batch")

    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        trust_remote_code=True,
    ).cuda().eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"expected {EXPECTED_PARAMETERS:,} parameters; found {parameter_count:,}"
        )
    if model.config.attention_space != args.expected_space:
        raise RuntimeError(
            f"expected attention_space={args.expected_space}; "
            f"found {model.config.attention_space}"
        )
    if model.config.attention_backend != "flash":
        raise RuntimeError(
            f"expected attention_backend=flash; found {model.config.attention_backend}"
        )

    with torch.inference_mode():
        output = model(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
        )
    expected_shape = (*batch["input_ids"].shape, model.config.vocab_size)
    if tuple(output.logits.shape) != expected_shape:
        raise RuntimeError(
            f"expected logits shape {expected_shape}; got {tuple(output.logits.shape)}"
        )
    valid = batch["attention_mask"].cuda().bool().unsqueeze(-1).expand_as(output.logits)
    if not bool(torch.isfinite(output.logits[valid]).all()):
        raise RuntimeError("non-finite logits on valid tokens")

    print(
        json.dumps(
            {
                "status": "ok",
                "device": device_name,
                "model": str(args.model.resolve()),
                "attention_space": model.config.attention_space,
                "attention_backend": model.config.attention_backend,
                "parameters": parameter_count,
                "dtype": str(next(model.parameters()).dtype),
                "padded_batch_shape": list(batch["input_ids"].shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
