#!/usr/bin/env python3
"""Aggregate the paired three-seed external LRA-CIFAR10 evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


VARIANTS = ("multispace-flash", "real-flash")
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "macro_f1", "loss", "examples_per_second")
T_CRITICAL_DF2_95 = 4.3026527297


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def paired_summary(values: list[float]) -> dict[str, float | int | list[float]]:
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    half_width = T_CRITICAL_DF2_95 * std / math.sqrt(len(values))
    return {
        "n": len(values),
        "per_seed_multispace_minus_real": values,
        "mean": mean,
        "sample_std": std,
        "ci95_student_t": [mean - half_width, mean + half_width],
    }


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    reports: dict[str, dict[int, dict[str, Any]]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = args.input_root / "cifar10" / variant / f"seed-{seed}" / "final_results.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["model"]["checkpoint_parameters"] != 99_985_152:
                raise RuntimeError(f"parameter mismatch in {path}")
            if report["seed"] != seed or report["model"]["variant"] != variant:
                raise RuntimeError(f"identity mismatch in {path}")
            reports[variant][seed] = report

    fingerprints = {
        report["dataset"]["fingerprint_sha256"]
        for by_seed in reports.values()
        for report in by_seed.values()
    }
    if len(fingerprints) != 1:
        raise RuntimeError(f"paired runs used different data: {sorted(fingerprints)}")
    for seed in SEEDS:
        left = reports["multispace-flash"][seed]
        right = reports["real-flash"][seed]
        if left["model"]["classifier_head_initialization_sha256"] != right["model"]["classifier_head_initialization_sha256"]:
            raise RuntimeError(f"paired head initialization differs for seed {seed}")
        if left["data_order_sha256"] != right["data_order_sha256"]:
            raise RuntimeError(f"paired data order differs for seed {seed}")
        if left["hyperparameters"] != right["hyperparameters"]:
            raise RuntimeError(f"paired hyperparameters differ for seed {seed}")
        if left["adaptation"] != right["adaptation"]:
            raise RuntimeError(f"paired adaptation differs for seed {seed}")

    variant_summaries: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_summaries[variant] = {
            metric: summary([float(reports[variant][seed]["test"][metric]) for seed in SEEDS])
            for metric in METRICS
        }
    for metric in METRICS:
        deltas = [
            float(reports["multispace-flash"][seed]["test"][metric])
            - float(reports["real-flash"][seed]["test"][metric])
            for seed in SEEDS
        ]
        paired[metric] = paired_summary(deltas)

    payload = {
        "schema_version": 1,
        "benchmark": "Long Range Arena",
        "task": "Image / sequential grayscale CIFAR-10",
        "reporting_class": protocol["reporting_class"],
        "protocol": protocol,
        "fairness_checks": {
            "data_fingerprint_sha256": fingerprints.pop(),
            "checkpoint_parameters_each": 99_985_152,
            "paired_head_initialization": True,
            "paired_data_order": True,
            "paired_hyperparameters": True,
            "paired_adaptation": True,
        },
        "seeds": list(SEEDS),
        "variants": variant_summaries,
        "paired_multispace_minus_real": paired,
        "per_seed": {
            variant: {
                str(seed): {metric: reports[variant][seed]["test"][metric] for metric in METRICS}
                for seed in SEEDS
            }
            for variant in VARIANTS
        },
        "suite_average": None,
        "suite_average_reason": "only one canonical LRA task is runnable under the declared protocol",
    }
    atomic_text(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    accuracy = paired["accuracy"]
    lines = [
        "# External paired LRA evaluation",
        "",
        "These results are not LRA-leaderboard comparable; see the protocol manifest.",
        "",
        "| Variant | Accuracy mean | Accuracy std | Macro-F1 mean | Throughput ex/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = variant_summaries[variant]
        lines.append(
            f"| {variant} | {item['accuracy']['mean']:.6f} | "
            f"{item['accuracy']['sample_std']:.6f} | {item['macro_f1']['mean']:.6f} | "
            f"{item['examples_per_second']['mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Paired accuracy delta (multispace - real): **{accuracy['mean']:.6f}** "
            f"(95% t interval {accuracy['ci95_student_t'][0]:.6f} to "
            f"{accuracy['ci95_student_t'][1]:.6f}; n=3 seeds).",
            "",
            "No six-task LRA average is reported because unsupported tasks were not truncated or silently resized.",
            "",
        ]
    )
    atomic_text(args.output.with_suffix(".md"), "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
