#!/usr/bin/env python3
"""Paired HANS and PAWS transfer evaluation for the exact 100M models.

The classifiers are the official BabyLM fine-tuning wrapper checkpoints:
MNLI transfers to HANS and QQP transfers to PAWS-Wiki.  Both attention
variants see exactly the same rows, tokenization, padding, and ordering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast

from evaluation_pipeline.finetune.classifier_model import (
    ModelForSequenceClassification,
)


EXPECTED_BASE_PARAMETERS = 99_985_152
EXPECTED_HANS_ROWS = 30_000
EXPECTED_PAWS_ROWS = 8_000
EXPECTED_HANS_COMMIT = "7299f6f657089ce06a0f98e7e81f8d0f5b7741ce"
EXPECTED_HANS_SHA256 = (
    "c55b62feef9913070e88f38938dc2492018c945ac81f70139346472494124e79"
)
EXPECTED_PAWS_SHA256 = (
    "ae342ff12bb84b84b95f468abf5db6cb7c7bd578271299fe9c99be75b8132f4d"
)
PAWS_DATA_REVISION = "254f2df8677712ee0972ddb92870dc79fed48aec"
MODEL_NAMES = ("multispace", "real")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required file is missing or empty: {path}")
    return path


def check_gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("HANS/PAWS model evaluation must run on CUDA")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name.upper():
        raise RuntimeError(f"HANS/PAWS evaluation requires an A100; found {name}")
    return {
        "device": name,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def load_hans(path: Path, repo: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = require_file(path)
    repo = repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_HANS_COMMIT:
        raise RuntimeError(
            f"HANS repo must be pinned at {EXPECTED_HANS_COMMIT}; found {commit}"
        )
    digest = sha256_file(path)
    if digest != EXPECTED_HANS_SHA256:
        raise RuntimeError(
            f"HANS data hash mismatch: expected {EXPECTED_HANS_SHA256}, found {digest}"
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "gold_label",
            "sentence1",
            "sentence2",
            "pairID",
            "heuristic",
            "subcase",
            "template",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"HANS data is missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            label_name = row["gold_label"]
            if label_name not in ("entailment", "non-entailment"):
                raise RuntimeError(f"invalid HANS label on row {row_index}: {label_name}")
            records.append(
                {
                    "row_index": row_index,
                    "id": row["pairID"],
                    "text_a": row["sentence1"],
                    "text_b": row["sentence2"],
                    "gold": 0 if label_name == "entailment" else 1,
                    "gold_label": label_name,
                    "heuristic": row["heuristic"],
                    "subcase": row["subcase"],
                    "template": row["template"],
                }
            )
    if len(records) != EXPECTED_HANS_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_HANS_ROWS:,} HANS rows; found {len(records):,}"
        )
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError("HANS pairID values are not unique")
    return records, {
        "repository": "https://github.com/tommccoy1/hans",
        "repository_path": str(repo),
        "commit": commit,
        "path": str(path),
        "sha256": digest,
        "total_rows": len(records),
    }


def load_paws(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = require_file(path)
    digest = sha256_file(path)
    if digest != EXPECTED_PAWS_SHA256:
        raise RuntimeError(
            f"PAWS data hash mismatch: expected {EXPECTED_PAWS_SHA256}, found {digest}"
        )
    table = pq.read_table(path, columns=["id", "sentence1", "sentence2", "label"])
    payload = table.to_pydict()
    records = []
    for row_index, (example_id, sentence1, sentence2, label) in enumerate(
        zip(
            payload["id"],
            payload["sentence1"],
            payload["sentence2"],
            payload["label"],
            strict=True,
        )
    ):
        label = int(label)
        if label not in (0, 1):
            raise RuntimeError(f"invalid PAWS label on row {row_index}: {label}")
        records.append(
            {
                "row_index": row_index,
                "id": str(example_id),
                "text_a": sentence1,
                "text_b": sentence2,
                "gold": label,
                "gold_label": "duplicate" if label == 1 else "not_duplicate",
            }
        )
    if len(records) != EXPECTED_PAWS_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_PAWS_ROWS:,} PAWS rows; found {len(records):,}"
        )
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError("PAWS ids are not unique")
    return records, {
        "repository": "https://github.com/google-research-datasets/paws",
        "source": "paws/labeled_final test split",
        "huggingface_parquet_revision": PAWS_DATA_REVISION,
        "path": str(path),
        "sha256": digest,
        "total_rows": len(records),
    }


def deterministic_subset(
    records: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        raise ValueError("--limit must be positive")
    if limit == 1:
        return [records[0]]
    indices = [round(i * (len(records) - 1) / (limit - 1)) for i in range(limit)]
    if len(set(indices)) != limit:
        raise RuntimeError("deterministic subset produced duplicate indices")
    return [records[index] for index in indices]


def load_tokenizer(path: Path):
    kwargs = {"trust_remote_code": True, "padding_side": "left"}
    try:
        return AutoProcessor.from_pretrained(str(path), **kwargs)
    except (ValueError, OSError, KeyError):
        try:
            return AutoTokenizer.from_pretrained(str(path), **kwargs)
        except (ValueError, OSError, KeyError):
            return PreTrainedTokenizerFast.from_pretrained(
                str(path), padding_side="left"
            )


def validate_manifest(
    manifest_path: Path,
    checkpoint: Path,
    base_model: Path,
    task: str,
    attention_space: str,
) -> dict[str, Any]:
    manifest_path = require_file(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_labels = (
        {"0": "entailment", "1": "neutral", "2": "contradiction"}
        if task == "mnli"
        else {"0": "not_duplicate", "1": "duplicate"}
    )
    checks = {
        "task": task,
        "attention_space": attention_space,
        "base_model_parameters": EXPECTED_BASE_PARAMETERS,
        "state_dict_format": "torch_state_dict",
        "wrapper_class": (
            "evaluation_pipeline.finetune.classifier_model."
            "ModelForSequenceClassification"
        ),
        "finetune_attention_backend": "torch",
        "pooling": "final_non_padding_token_with_left_padding",
        "id2label": expected_labels,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"manifest {manifest_path} has {key}={payload.get(key)!r}; "
                f"expected {expected!r}"
            )
    if Path(payload["state_dict_path"]).resolve() != checkpoint.resolve():
        raise RuntimeError(f"manifest checkpoint path does not match {checkpoint}")
    if Path(payload["base_model_path"]).resolve() != base_model.resolve():
        raise RuntimeError(f"manifest base-model path does not match {base_model}")
    return payload


def classifier_args(base_model: Path, num_labels: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_name_or_path=str(base_model),
        revision_name=None,
        enc_dec=False,
        three_d_triangular_causal_mask=False,
        classifier_layer_norm_eps=1.0e-5,
        classifier_dropout=0.1,
        num_labels=num_labels,
        take_final=True,
    )


def load_classifier(
    base_model: Path,
    checkpoint: Path,
    num_labels: int,
    expected_space: str,
) -> tuple[ModelForSequenceClassification, dict[str, Any]]:
    checkpoint = require_file(checkpoint)
    model = ModelForSequenceClassification(classifier_args(base_model, num_labels))
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    base_parameters = sum(p.numel() for p in model.transformer.parameters())
    if base_parameters != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(
            f"expected {EXPECTED_BASE_PARAMETERS:,} encoder parameters; "
            f"found {base_parameters:,}"
        )
    config = model.transformer.config
    if getattr(config, "attention_space", None) != expected_space:
        raise RuntimeError(
            f"expected attention_space={expected_space}; "
            f"found {getattr(config, 'attention_space', None)}"
        )
    backends = list(getattr(config, "attention_backends", []))
    if getattr(config, "attention_backend", None) != "torch" or any(
        backend != "torch" for backend in backends
    ):
        raise RuntimeError("classification compatibility must select torch attention")
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("classifier evaluation must retain official FP32 weights")

    model.cuda().eval()
    return model, {
        "base_model": str(base_model.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "attention_space": expected_space,
        "finetune_and_eval_backend": "torch",
        "dtype": str(next(model.parameters()).dtype),
        "base_parameters": base_parameters,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.inference_mode()
def predict(
    model: ModelForSequenceClassification,
    tokenizer,
    records: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> tuple[list[int], list[list[float]], dict[str, float]]:
    batches = list(batched(records, batch_size))
    if not batches:
        raise RuntimeError("cannot evaluate an empty dataset")

    def run_batch(batch: Sequence[dict[str, Any]]) -> torch.Tensor:
        texts = [(record["text_a"], record["text_b"]) for record in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.cuda(non_blocking=True)
        attention_mask = encoded.attention_mask.cuda(non_blocking=True)
        return model(input_ids, attention_mask)

    # Exclude one-time CUDA/library initialization from throughput.
    warmup_logits = run_batch(batches[0])
    torch.cuda.synchronize()
    del warmup_logits
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions: list[int] = []
    logits_out: list[list[float]] = []
    for batch in batches:
        logits = run_batch(batch)
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        logits_out.extend(logits.float().cpu().tolist())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if len(predictions) != len(records):
        raise RuntimeError("prediction row count differs from evaluation row count")
    return predictions, logits_out, {
        "seconds": elapsed,
        "examples_per_second": len(records) / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def confusion(gold: Sequence[int], pred: Sequence[int]) -> dict[str, int]:
    tn = fp = fn = tp = 0
    for target, output in zip(gold, pred, strict=True):
        if target == 1 and output == 1:
            tp += 1
        elif target == 0 and output == 1:
            fp += 1
        elif target == 1 and output == 0:
            fn += 1
        else:
            tn += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def binary_metrics(gold: Sequence[int], pred: Sequence[int]) -> dict[str, Any]:
    counts = confusion(gold, pred)
    tn, fp, fn, tp = (counts[key] for key in ("tn", "fp", "fn", "tp"))
    total = len(gold)
    accuracy = (tp + tn) / total
    f1_denominator = 2 * tp + fp + fn
    f1 = 0.0 if f1_denominator == 0 else 2 * tp / f1_denominator
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = 0.0 if mcc_denominator == 0 else (tp * tn - fp * fn) / mcc_denominator
    return {
        "row_count": total,
        "accuracy": accuracy,
        "f1": f1,
        "mcc": mcc,
        "confusion": counts,
    }


def accuracy(gold: Sequence[int], pred: Sequence[int]) -> float:
    return sum(a == b for a, b in zip(gold, pred, strict=True)) / len(gold)


def grouped_hans_metrics(
    records: Sequence[dict[str, Any]], pred: Sequence[int]
) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[int]]] = {
        "by_heuristic": defaultdict(list),
        "by_subcase": defaultdict(list),
        "by_template": defaultdict(list),
        "by_heuristic_and_gold_label": defaultdict(list),
        "by_subcase_and_gold_label": defaultdict(list),
    }
    for index, record in enumerate(records):
        buckets["by_heuristic"][record["heuristic"]].append(index)
        buckets["by_subcase"][record["subcase"]].append(index)
        buckets["by_template"][record["template"]].append(index)
        buckets["by_heuristic_and_gold_label"][
            f"{record['heuristic']}::{record['gold_label']}"
        ].append(index)
        buckets["by_subcase_and_gold_label"][
            f"{record['subcase']}::{record['gold_label']}"
        ].append(index)
    result: dict[str, Any] = {}
    for group_name, groups in buckets.items():
        result[group_name] = {}
        for key in sorted(groups):
            indices = groups[key]
            group_gold = [records[index]["gold"] for index in indices]
            group_pred = [pred[index] for index in indices]
            result[group_name][key] = {
                "row_count": len(indices),
                "accuracy": accuracy(group_gold, group_pred),
            }
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_predictions(
    path: Path,
    task: str,
    records: Sequence[dict[str, Any]],
    raw_predictions: Sequence[int],
    logits: Sequence[Sequence[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, raw_pred, row_logits in zip(
            records, raw_predictions, logits, strict=True
        ):
            if task == "hans":
                binary_pred = 0 if raw_pred == 0 else 1
                pred_label = "entailment" if binary_pred == 0 else "non-entailment"
                extra = {
                    "mnli_prediction": raw_pred,
                    "mnli_prediction_label": (
                        "entailment", "neutral", "contradiction"
                    )[raw_pred],
                    "heuristic": record["heuristic"],
                    "subcase": record["subcase"],
                    "template": record["template"],
                }
            else:
                binary_pred = raw_pred
                pred_label = "duplicate" if binary_pred == 1 else "not_duplicate"
                extra = {"qqp_prediction": raw_pred}
            payload = {
                "row_index": record["row_index"],
                "id": record["id"],
                "gold": record["gold"],
                "gold_label": record["gold_label"],
                "prediction": binary_pred,
                "prediction_label": pred_label,
                "logits": [float(value) for value in row_logits],
                **extra,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_hans_official_predictions(
    path: Path,
    records: Sequence[dict[str, Any]],
    predictions: Sequence[int],
) -> None:
    """Write the two-column format consumed by HANS's official scorer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("pairID", "gold_label"))
        for record, prediction in zip(records, predictions, strict=True):
            writer.writerow(
                (
                    record["id"],
                    "entailment" if prediction == 0 else "non-entailment",
                )
            )


def comparison(
    task: str,
    records: Sequence[dict[str, Any]],
    predictions: dict[str, Sequence[int]],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gold = [record["gold"] for record in records]
    ms = predictions["multispace"]
    real = predictions["real"]
    ms_correct = [target == output for target, output in zip(gold, ms, strict=True)]
    real_correct = [target == output for target, output in zip(gold, real, strict=True)]
    ms_only = sum(a and not b for a, b in zip(ms_correct, real_correct, strict=True))
    real_only = sum(b and not a for a, b in zip(ms_correct, real_correct, strict=True))
    both_correct = sum(a and b for a, b in zip(ms_correct, real_correct, strict=True))
    both_wrong = len(records) - ms_only - real_only - both_correct
    result: dict[str, Any] = {
        "row_count": len(records),
        "identical_row_ids": True,
        "prediction_disagreements": sum(a != b for a, b in zip(ms, real, strict=True)),
        "correctness": {
            "multispace_only_correct": ms_only,
            "real_only_correct": real_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "accuracy_delta_multispace_minus_real": (
            metrics["multispace"]["accuracy"] - metrics["real"]["accuracy"]
        ),
    }
    if task == "paws":
        result["f1_delta_multispace_minus_real"] = (
            metrics["multispace"]["f1"] - metrics["real"]["f1"]
        )
        result["mcc_delta_multispace_minus_real"] = (
            metrics["multispace"]["mcc"] - metrics["real"]["mcc"]
        )
    return result


def evaluate_task(
    task: str,
    records: list[dict[str, Any]],
    source: dict[str, Any],
    args: argparse.Namespace,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    selected = deterministic_subset(records, args.limit)
    task_dir = args.output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    classifier_task = "mnli" if task == "hans" else "qqp"
    num_labels = 3 if task == "hans" else 2
    gold = [record["gold"] for record in selected]

    base_models = {
        "multispace": args.multispace_base,
        "real": args.real_base,
    }
    checkpoints = {
        "multispace": args.multispace_mnli if task == "hans" else args.multispace_qqp,
        "real": args.real_mnli if task == "hans" else args.real_qqp,
    }
    tokenizer_hashes = {
        name: sha256_file(require_file(path / "tokenizer.json"))
        for name, path in base_models.items()
    }
    if len(set(tokenizer_hashes.values())) != 1:
        raise RuntimeError(f"paired model tokenizer hashes differ: {tokenizer_hashes}")
    tokenizer = load_tokenizer(base_models["multispace"])
    if getattr(tokenizer, "padding_side", None) != "left":
        raise RuntimeError("tokenizer must use left padding for take_final pooling")

    all_predictions: dict[str, list[int]] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        base_model = base_models[model_name].resolve()
        checkpoint = checkpoints[model_name].resolve()
        manifest_path = checkpoint.parent / "checkpoint_manifest.json"
        manifest = validate_manifest(
            manifest_path,
            checkpoint,
            base_model,
            classifier_task,
            model_name,
        )
        model, metadata = load_classifier(
            base_model, checkpoint, num_labels, model_name
        )
        raw_pred, logits, performance = predict(
            model, tokenizer, selected, args.batch_size, args.max_length
        )
        binary_pred = (
            [0 if prediction == 0 else 1 for prediction in raw_pred]
            if task == "hans"
            else raw_pred
        )
        if task == "hans":
            task_metrics: dict[str, Any] = {
                "row_count": len(selected),
                "accuracy": accuracy(gold, binary_pred),
                **grouped_hans_metrics(selected, binary_pred),
            }
        else:
            task_metrics = binary_metrics(gold, binary_pred)
        task_metrics["performance"] = performance
        all_predictions[model_name] = binary_pred
        all_metrics[model_name] = task_metrics
        model_metadata[model_name] = {
            **metadata,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "validation_result_path": manifest["validation_result_path"],
        }
        write_predictions(
            task_dir / f"{model_name}_predictions.jsonl",
            task,
            selected,
            raw_pred,
            logits,
        )
        if task == "hans":
            write_hans_official_predictions(
                task_dir / f"{model_name}_official_predictions.csv",
                selected,
                binary_pred,
            )
        del model, logits, raw_pred
        torch.cuda.empty_cache()

    if len(all_predictions["multispace"]) != len(all_predictions["real"]):
        raise RuntimeError("paired prediction lengths differ")
    summary = {
        "status": "complete",
        "evaluation": (
            "MNLI-to-HANS adversarial transfer"
            if task == "hans"
            else "QQP-to-PAWS-Wiki adversarial transfer"
        ),
        "task": task,
        "protocol": "paired_rows_and_tokenizer",
        "preflight_subset": args.limit is not None,
        "evaluated_rows": len(selected),
        "selected_row_indices_sha256": hashlib.sha256(
            json.dumps([row["row_index"] for row in selected]).encode("utf-8")
        ).hexdigest(),
        "source": source,
        "hardware": hardware,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "pooling": "final_non_padding_token_with_left_padding",
        "tokenizer_sha256": tokenizer_hashes,
        "models": model_metadata,
        "metrics": all_metrics,
        "comparison": comparison(task, selected, all_predictions, all_metrics),
        "prediction_files": {
            name: {
                "jsonl": str((task_dir / f"{name}_predictions.jsonl").resolve()),
                **(
                    {
                        "hans_official_csv": str(
                            (
                                task_dir
                                / f"{name}_official_predictions.csv"
                            ).resolve()
                        )
                    }
                    if task == "hans"
                    else {}
                ),
            }
            for name in MODEL_NAMES
        },
    }
    write_json(task_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("hans", "paws", "both"))
    parser.add_argument("--hans-repo", required=True, type=Path)
    parser.add_argument("--hans-data", required=True, type=Path)
    parser.add_argument("--paws-data", required=True, type=Path)
    parser.add_argument("--multispace-base", required=True, type=Path)
    parser.add_argument("--real-base", required=True, type=Path)
    parser.add_argument("--multispace-mnli", required=True, type=Path)
    parser.add_argument("--real-mnli", required=True, type=Path)
    parser.add_argument("--multispace-qqp", required=True, type=Path)
    parser.add_argument("--real-qqp", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-length", default=512, type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    return args


def main() -> None:
    args = parse_args()
    hardware = check_gpu()
    summaries = {}
    if args.task in ("hans", "both"):
        hans_records, hans_source = load_hans(args.hans_data, args.hans_repo)
        summaries["hans"] = evaluate_task(
            "hans", hans_records, hans_source, args, hardware
        )
    if args.task in ("paws", "both"):
        paws_records, paws_source = load_paws(args.paws_data)
        summaries["paws"] = evaluate_task(
            "paws", paws_records, paws_source, args, hardware
        )
    write_json(args.output_dir / "run_summary.json", summaries)


if __name__ == "__main__":
    main()
