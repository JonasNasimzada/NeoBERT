#!/usr/bin/env python3
"""Aggregate paired three-seed SuperGLUE reports with fairness guards."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch


TASKS = ("boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc")
VARIANTS = ("multispace", "real")
EXPECTED_BASE_PARAMETERS = 99_985_152


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    args = parser.parse_args()

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("SuperGLUE aggregation is required to run in the A100 chain")

    reports: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        task: {variant: {} for variant in VARIANTS} for task in TASKS
    }
    fairness = []
    for task in TASKS:
        for seed in args.seeds:
            pair = []
            for variant in VARIANTS:
                path = args.input_root / task / variant / f"seed-{seed}" / "report.json"
                if not path.is_file():
                    raise FileNotFoundError(f"missing full SuperGLUE report: {path}")
                report = json.loads(path.read_text(encoding="utf-8"))
                expected = {"status": "ok", "task": task, "variant": variant, "seed": seed}
                for key, value in expected.items():
                    if report.get(key) != value:
                        raise RuntimeError(
                            f"{path} has {key}={report.get(key)!r}; expected {value!r}"
                        )
                if report["model_metadata"]["base_parameters"] != EXPECTED_BASE_PARAMETERS:
                    raise RuntimeError(f"parameter mismatch in {path}")
                if report["protocol"]["preflight"]:
                    raise RuntimeError(f"preflight report found in full results: {path}")
                reports[task][variant][seed] = report
                pair.append(report)
            left, right = pair
            if left["task_head_initialization_sha256"] != right["task_head_initialization_sha256"]:
                raise RuntimeError(f"unpaired task-head initialization for {task}/seed-{seed}")
            if left["protocol_sha256"] != right["protocol_sha256"]:
                raise RuntimeError(f"unpaired protocol for {task}/seed-{seed}")
            if left["train_rows"] != right["train_rows"] or left["validation_rows"] != right["validation_rows"]:
                raise RuntimeError(f"unpaired row counts for {task}/seed-{seed}")
            fairness.append(
                {
                    "task": task,
                    "seed": seed,
                    "task_head_initialization_sha256": left[
                        "task_head_initialization_sha256"
                    ],
                    "protocol_sha256": left["protocol_sha256"],
                    "train_rows": left["train_rows"],
                    "validation_rows": left["validation_rows"],
                }
            )

    task_summary: dict[str, Any] = {}
    for task in TASKS:
        variant_summary = {}
        for variant in VARIANTS:
            values = [reports[task][variant][seed]["primary_score"] for seed in args.seeds]
            metric_names = sorted(reports[task][variant][args.seeds[0]]["metrics"])
            metric_summary = {
                name: mean_std(
                    [reports[task][variant][seed]["metrics"][name] for seed in args.seeds]
                )
                for name in metric_names
            }
            variant_summary[variant] = {
                "primary_score": mean_std(values),
                "metrics": metric_summary,
                "per_seed_primary_score": dict(zip(map(str, args.seeds), values, strict=True)),
            }
        deltas = [
            reports[task]["multispace"][seed]["primary_score"]
            - reports[task]["real"][seed]["primary_score"]
            for seed in args.seeds
        ]
        task_summary[task] = {
            **variant_summary,
            "paired_delta_multispace_minus_real": {
                **mean_std(deltas),
                "per_seed": dict(zip(map(str, args.seeds), deltas, strict=True)),
            },
        }

    overall_per_seed: dict[str, dict[int, float]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in args.seeds:
            overall_per_seed[variant][seed] = statistics.mean(
                reports[task][variant][seed]["primary_score"] for task in TASKS
            )
    overall_deltas = [
        overall_per_seed["multispace"][seed] - overall_per_seed["real"][seed]
        for seed in args.seeds
    ]
    overall = {
        variant: {
            **mean_std(list(overall_per_seed[variant].values())),
            "per_seed": {
                str(seed): overall_per_seed[variant][seed] for seed in args.seeds
            },
        }
        for variant in VARIANTS
    }
    overall["paired_delta_multispace_minus_real"] = {
        **mean_std(overall_deltas),
        "per_seed": dict(zip(map(str, args.seeds), overall_deltas, strict=True)),
    }

    diagnostics: dict[str, Any] = {}
    for diagnostic in ("axb", "axg"):
        metric_name = "matthews_correlation" if diagnostic == "axb" else "accuracy"
        diagnostics[diagnostic] = {}
        for variant in VARIANTS:
            values = [
                reports["rte"][variant][seed]["diagnostics"][diagnostic]["metrics"][metric_name]
                for seed in args.seeds
            ]
            diagnostics[diagnostic][variant] = mean_std(values)
        deltas = [
            reports["rte"]["multispace"][seed]["diagnostics"][diagnostic]["metrics"][metric_name]
            - reports["rte"]["real"][seed]["diagnostics"][diagnostic]["metrics"][metric_name]
            for seed in args.seeds
        ]
        diagnostics[diagnostic]["paired_delta_multispace_minus_real"] = mean_std(deltas)
        diagnostics[diagnostic]["metric"] = metric_name
        diagnostics[diagnostic]["included_in_primary_aggregate"] = False

    result = {
        "status": "ok",
        "benchmark": "SuperGLUE",
        "evaluation_split": "official validation",
        "leaderboard_submission": False,
        "seeds": args.seeds,
        "tasks": task_summary,
        "overall": overall,
        "diagnostics": diagnostics,
        "fairness_checks": {
            "status": "passed",
            "exact_base_parameters": EXPECTED_BASE_PARAMETERS,
            "paired_task_head_and_protocol_records": fairness,
        },
        "score_definition": (
            "unweighted mean of eight task scores; CB, MultiRC, and ReCoRD each "
            "first average their two official metrics"
        ),
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "summary.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Paired SuperGLUE validation results",
        "",
        "All values are proportions. Delta is multispace minus real.",
        "",
        "| Task | Multispace mean ± sd | Real mean ± sd | Paired delta mean ± sd |",
        "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        summary = task_summary[task]
        lines.append(
            f"| {task} | {summary['multispace']['primary_score']['mean']:.4f} ± "
            f"{summary['multispace']['primary_score']['std']:.4f} | "
            f"{summary['real']['primary_score']['mean']:.4f} ± "
            f"{summary['real']['primary_score']['std']:.4f} | "
            f"{summary['paired_delta_multispace_minus_real']['mean']:+.4f} ± "
            f"{summary['paired_delta_multispace_minus_real']['std']:.4f} |"
        )
    lines.extend(
        [
            f"| **Overall** | **{overall['multispace']['mean']:.4f} ± {overall['multispace']['std']:.4f}** | "
            f"**{overall['real']['mean']:.4f} ± {overall['real']['std']:.4f}** | "
            f"**{overall['paired_delta_multispace_minus_real']['mean']:+.4f} ± "
            f"{overall['paired_delta_multispace_minus_real']['std']:.4f}** |",
            "",
            "AX-b and AX-g are RTE-transfer diagnostics and are excluded from Overall.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
