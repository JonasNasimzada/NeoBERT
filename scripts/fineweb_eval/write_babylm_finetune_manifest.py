#!/usr/bin/env python3
"""Write reproducibility and label metadata for a BabyLM classifier state dict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "boolq": ("false", "true"),
    "multirc": ("false", "true"),
    "rte": ("entailment", "not_entailment"),
    "wsc": ("false", "true"),
    "mrpc": ("not_equivalent", "equivalent"),
    "qqp": ("not_duplicate", "duplicate"),
    "mnli": ("entailment", "neutral", "contradiction"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=tuple(LABELS))
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--metric-for-valid", required=True)
    args = parser.parse_args()

    for path in (args.base_model / "config.json", args.checkpoint, args.result):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required fine-tune artifact is missing: {path}")

    base_config = json.loads((args.base_model / "config.json").read_text(encoding="utf-8"))
    labels = LABELS[args.task]
    hidden_size = int(base_config["hidden_size"])
    head_parameters = (
        hidden_size * hidden_size
        + hidden_size
        + hidden_size * len(labels)
        + len(labels)
    )
    payload = {
        "evaluation_status": "external_diagnostic_not_babylm_leaderboard_eligible",
        "task": args.task,
        "id2label": {str(index): label for index, label in enumerate(labels)},
        "label2id": {label: index for index, label in enumerate(labels)},
        "num_labels": len(labels),
        "state_dict_format": "torch_state_dict",
        "wrapper_class": (
            "evaluation_pipeline.finetune.classifier_model."
            "ModelForSequenceClassification"
        ),
        "state_dict_path": str(args.checkpoint.resolve()),
        "base_model_path": str(args.base_model.resolve()),
        "base_model_parameters": 99_985_152,
        "classifier_head_parameters": head_parameters,
        "total_trainable_parameters": 99_985_152 + head_parameters,
        "attention_space": base_config["attention_space"],
        "checkpoint_attention_backend": base_config["attention_backend"],
        "finetune_attention_backend": "torch",
        "backend_note": (
            "The learned tensors are unchanged. The official padded FP32 "
            "fine-tuning batches use the mathematically equivalent PyTorch "
            "attention implementation instead of the padding-free Flash kernel."
        ),
        "pooling": "final_non_padding_token_with_left_padding",
        "classifier_initialization": {
            "seed": args.seed,
            "paired": True,
            "protocol": (
                "restore CPU/CUDA RNG after variant-specific encoder loading "
                "before constructing the official classifier head"
            ),
        },
        "evaluator": {
            "repository": "https://github.com/babylm-org/babylm-eval",
            "commit": "6f825c291e2c4c78ad33b1935fd64d45f52642dc",
            "core_data_revision": "8d52da9424a9ff30b9e8266c4f751aba9c504233",
        },
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "sequence_length": 512,
            "valid_batch_size": 64,
            "weight_decay": 0.01,
            "warmup_proportion": 0.06,
            "scheduler": "cosine",
            "metric_for_valid": args.metric_for_valid,
            "keep_best_model": True,
        },
        "validation_result_path": str(args.result.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
