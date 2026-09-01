#!/usr/bin/env python3
"""Aggregate the paired 3-seed GLUE public-validation experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


TASKS = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli")
SEEDS = (42, 43, 44)
VARIANTS = ("multispace-flash", "real-flash")


def summary(values: list[float]) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }
    return result


def paired_summary(values: list[float]) -> dict[str, float | int]:
    result = summary(values)
    if len(values) > 1:
        half_width = float(t.ppf(0.975, len(values) - 1)) * float(
            result["std"]
        ) / math.sqrt(len(values))
    else:
        half_width = 0.0
    result["ci95_low"] = float(result["mean"]) - half_width
    result["ci95_high"] = float(result["mean"]) + half_width
    result["wins"] = sum(value > 0 for value in values)
    result["ties"] = sum(value == 0 for value in values)
    result["losses"] = sum(value < 0 for value in values)
    return result


def flatten_metrics(report: dict) -> dict[str, float]:
    flattened = {}
    for split, split_report in report["validation"].items():
        for metric, value in split_report["metrics"].items():
            flattened[f"{split}/{metric}"] = float(value)
    return flattened


def load_report(root: Path, variant: str, seed: int, task: str) -> dict:
    path = root / variant / f"seed-{seed}" / task / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "variant": variant,
        "seed": seed,
        "task": task,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(
                f"invalid {key} in {path}: {report.get(key)!r} != {value!r}"
            )
    if report["model"]["base_parameters"] != 99_985_152:
        raise RuntimeError(f"wrong base parameter count: {path}")
    if "A100" not in report["hardware"]["device"].upper():
        raise RuntimeError(f"non-A100 result rejected: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = {
        (variant, seed, task): load_report(args.root, variant, seed, task)
        for variant in VARIANTS
        for seed in SEEDS
        for task in TASKS
    }

    paired_checks = []
    for seed in SEEDS:
        for task in TASKS:
            multispace = reports[("multispace-flash", seed, task)]
            real = reports[("real-flash", seed, task)]
            comparisons = {
                "head_initialization_sha256": (
                    multispace["pairing"]["head_initialization_sha256"],
                    real["pairing"]["head_initialization_sha256"],
                ),
                "data_order_sha256": (
                    multispace["pairing"]["data_order_sha256"],
                    real["pairing"]["data_order_sha256"],
                ),
                "hyperparameters_sha256": (
                    multispace["hyperparameters_sha256"],
                    real["hyperparameters_sha256"],
                ),
                "data_manifest_sha256": (
                    multispace["dataset"]["manifest_sha256"],
                    real["dataset"]["manifest_sha256"],
                ),
                "tokenizer_sha256": (
                    multispace["model"]["tokenizer_sha256"],
                    real["model"]["tokenizer_sha256"],
                ),
                "total_steps": (
                    multispace["optimization"]["total_steps"],
                    real["optimization"]["total_steps"],
                ),
            }
            mismatches = [
                key for key, (left, right) in comparisons.items() if left != right
            ]
            if mismatches:
                raise RuntimeError(
                    f"unpaired {task}/seed-{seed}: {', '.join(mismatches)}"
                )
            paired_checks.append(
                {
                    "task": task,
                    "seed": seed,
                    "status": "identical",
                    **{key: values[0] for key, values in comparisons.items()},
                }
            )

    task_results = {}
    for task in TASKS:
        variant_scores = {
            variant: [
                float(reports[(variant, seed, task)]["glue_task_score"])
                for seed in SEEDS
            ]
            for variant in VARIANTS
        }
        deltas = [
            variant_scores["multispace-flash"][index]
            - variant_scores["real-flash"][index]
            for index in range(len(SEEDS))
        ]
        all_metric_names = sorted(
            flatten_metrics(reports[(VARIANTS[0], SEEDS[0], task)])
        )
        metric_results = {}
        for metric_name in all_metric_names:
            by_variant = {
                variant: [
                    flatten_metrics(reports[(variant, seed, task)])[metric_name]
                    for seed in SEEDS
                ]
                for variant in VARIANTS
            }
            metric_deltas = [
                by_variant["multispace-flash"][index]
                - by_variant["real-flash"][index]
                for index in range(len(SEEDS))
            ]
            metric_results[metric_name] = {
                "multispace": summary(by_variant["multispace-flash"]),
                "real": summary(by_variant["real-flash"]),
                "paired_delta_multispace_minus_real": paired_summary(metric_deltas),
                "per_seed": {
                    str(seed): {
                        "multispace": by_variant["multispace-flash"][index],
                        "real": by_variant["real-flash"][index],
                        "delta": metric_deltas[index],
                    }
                    for index, seed in enumerate(SEEDS)
                },
            }
        task_results[task] = {
            "glue_task_score": {
                "multispace": summary(variant_scores["multispace-flash"]),
                "real": summary(variant_scores["real-flash"]),
                "paired_delta_multispace_minus_real": paired_summary(deltas),
                "per_seed": {
                    str(seed): {
                        "multispace": variant_scores["multispace-flash"][index],
                        "real": variant_scores["real-flash"][index],
                        "delta": deltas[index],
                    }
                    for index, seed in enumerate(SEEDS)
                },
            },
            "official_metrics": metric_results,
        }

    overall_by_variant = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            overall_by_variant[variant].append(
                mean(
                    float(reports[(variant, seed, task)]["glue_task_score"])
                    for task in TASKS
                )
            )
    overall_deltas = [
        overall_by_variant["multispace-flash"][index]
        - overall_by_variant["real-flash"][index]
        for index in range(len(SEEDS))
    ]

    ax_records = {
        variant: {
            str(seed): reports[(variant, seed, "mnli")]["diagnostics"]["ax"]
            for seed in SEEDS
        }
        for variant in VARIANTS
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "GLUE public validation benchmark",
        "leaderboard_eligible": False,
        "seeds": list(SEEDS),
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "score_scale": "0_to_1; multiply by 100 for GLUE-style display",
        "overall_nine_task_macro_average": {
            "multispace": summary(overall_by_variant["multispace-flash"]),
            "real": summary(overall_by_variant["real-flash"]),
            "paired_delta_multispace_minus_real": paired_summary(overall_deltas),
            "per_seed": {
                str(seed): {
                    "multispace": overall_by_variant["multispace-flash"][index],
                    "real": overall_by_variant["real-flash"][index],
                    "delta": overall_deltas[index],
                }
                for index, seed in enumerate(SEEDS)
            },
        },
        "task_results": task_results,
        "paired_invariant_checks": paired_checks,
        "ax_diagnostic": {
            "status": "unscored_hidden_test_labels",
            "excluded_from_aggregate": True,
            "prediction_exports": ax_records,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)

    markdown = [
        "# Paired GLUE public-validation results",
        "",
        "Scores are means across seeds 42/43/44 and are not official test-server scores.",
        "",
        "| Task | Multispace | Real MHA | Paired delta |",
        "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        result = task_results[task]["glue_task_score"]
        markdown.append(
            f"| {task.upper()} | {100 * result['multispace']['mean']:.3f} | "
            f"{100 * result['real']['mean']:.3f} | "
            f"{100 * result['paired_delta_multispace_minus_real']['mean']:+.3f} |"
        )
    overall = payload["overall_nine_task_macro_average"]
    markdown.extend(
        [
            f"| **Macro average** | **{100 * overall['multispace']['mean']:.3f}** | "
            f"**{100 * overall['real']['mean']:.3f}** | "
            f"**{100 * overall['paired_delta_multispace_minus_real']['mean']:+.3f}** |",
            "",
            "AX predictions are exported from each MNLI run but remain unscored and are excluded from the aggregate.",
            "",
        ]
    )
    (args.output.parent / "summary.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
