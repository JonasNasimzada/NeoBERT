#!/usr/bin/env python3
"""A100 forward/backward preflight for the official BabyLM classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from evaluation_pipeline.finetune.classifier_model import ModelForSequenceClassification


EXPECTED_BASE_PARAMETERS = 99_985_152


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected-space", required=True, choices=("real", "multispace"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("BabyLM fine-tuning preflight must run on CUDA")
    device_name = torch.cuda.get_device_name()
    if "A100" not in device_name.upper():
        raise RuntimeError(f"fine-tuning preflight requires an A100; found {device_name}")

    classifier_args = SimpleNamespace(
        model_name_or_path=str(args.model),
        revision_name=None,
        enc_dec=False,
        three_d_triangular_causal_mask=False,
        classifier_layer_norm_eps=1.0e-5,
        classifier_dropout=0.1,
        num_labels=3,
        take_final=True,
    )
    model = ModelForSequenceClassification(classifier_args).cuda().train()
    head_digest = hashlib.sha256()
    for name, tensor in sorted(model.classifier.state_dict().items()):
        head_digest.update(f"{name}\0".encode("utf-8"))
        head_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    classifier_head_initialization_sha256 = head_digest.hexdigest()
    base_parameters = sum(
        parameter.numel() for parameter in model.transformer.parameters()
    )
    if base_parameters != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(
            f"expected {EXPECTED_BASE_PARAMETERS:,} base parameters; "
            f"found {base_parameters:,}"
        )
    config = model.transformer.config
    if config.attention_space != args.expected_space:
        raise RuntimeError(
            f"expected attention_space={args.expected_space}; found {config.attention_space}"
        )
    if config.attention_backend != "torch" \
        or any(backend != "torch" for backend in config.attention_backends):
        raise RuntimeError("fine-tuning compatibility did not select torch attention")
    if next(model.transformer.parameters()).dtype != torch.float32:
        raise RuntimeError("fine-tuning encoder must remain FP32")

    input_ids = torch.tensor(
        [
            [0, 0, 101, 2023, 2003, 1037, 3231, 102],
            [101, 2178, 3231, 2007, 2062, 19204, 2015, 102],
        ],
        dtype=torch.long,
        device="cuda",
    )
    attention_mask = input_ids.ne(0).long()
    labels = torch.tensor([0, 2], dtype=torch.long, device="cuda")
    logits = model(input_ids, attention_mask)
    if tuple(logits.shape) != (2, 3):
        raise RuntimeError(f"unexpected classifier logits shape: {tuple(logits.shape)}")
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    finite_gradients = [
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not finite_gradients or not all(finite_gradients):
        raise RuntimeError("fine-tuning preflight produced non-finite gradients")

    print(
        json.dumps(
            {
                "status": "ok",
                "device": device_name,
                "model": str(args.model.resolve()),
                "attention_space": config.attention_space,
                "checkpoint_backend": "flash",
                "finetune_backend": config.attention_backend,
                "base_parameters": base_parameters,
                "dtype": str(next(model.transformer.parameters()).dtype),
                "loss": float(loss.detach()),
                "classifier_head_initialization_sha256": (
                    classifier_head_initialization_sha256
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
