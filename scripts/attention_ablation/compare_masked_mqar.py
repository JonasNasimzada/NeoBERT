#!/usr/bin/env python3
"""Compare paired real-MHA and multispace masked-MQAR JSON reports."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


EXPECTED_PARAMETERS = 99_985_152
Z_95 = 1.959963984540054


def _load_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"masked-MQAR report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("benchmark") != "masked-mqar":
        raise ValueError(f"not a masked-MQAR report: {path}")
    return report


def _mean_confidence_interval(differences: Sequence[float]) -> dict[str, float | int]:
    if not differences:
        raise ValueError("paired confidence interval requires observations")
    count = len(differences)
    mean = sum(differences) / count
    if count == 1:
        standard_error = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
        standard_error = math.sqrt(variance / count)
    return {
        "paired_examples": count,
        "mean_difference": mean,
        "standard_error": standard_error,
        "ci95_low": mean - Z_95 * standard_error,
        "ci95_high": mean + Z_95 * standard_error,
    }


def _summary_delta(multispace: Mapping, real: Mapping) -> dict[str, float | int]:
    accuracy_key = "micro_accuracy" if "micro_accuracy" in multispace else "accuracy"
    nll_key = "micro_masked_nll" if "micro_masked_nll" in multispace else "masked_nll"
    return {
        "examples": int(multispace["examples"]),
        "multispace_accuracy": float(multispace[accuracy_key]),
        "real_accuracy": float(real[accuracy_key]),
        "accuracy_difference_multispace_minus_real": (
            float(multispace[accuracy_key]) - float(real[accuracy_key])
        ),
        "multispace_masked_nll": float(multispace[nll_key]),
        "real_masked_nll": float(real[nll_key]),
        "masked_nll_difference_multispace_minus_real": (
            float(multispace[nll_key]) - float(real[nll_key])
        ),
        "multispace_token_positions_per_second": float(
            multispace["token_positions_per_second"]
        ),
        "real_token_positions_per_second": float(real["token_positions_per_second"]),
        "real_over_multispace_throughput_ratio": (
            float(real["token_positions_per_second"])
            / float(multispace["token_positions_per_second"])
        ),
    }


def _group_deltas(multispace: Mapping, real: Mapping) -> dict[str, Mapping]:
    if set(multispace) != set(real):
        raise AssertionError("paired summary group keys differ")
    return {
        key: _summary_delta(multispace[key], real[key])
        for key in sorted(multispace)
    }


def compare_reports(multispace: Mapping, real: Mapping) -> dict:
    variants = {
        multispace.get("model", {}).get("variant"),
        real.get("model", {}).get("variant"),
    }
    if variants != {"multispace-flash", "real-flash"}:
        raise AssertionError(f"expected the 100M paired variants; got {variants!r}")
    if multispace["model"]["variant"] != "multispace-flash":
        raise AssertionError("--multispace report does not contain multispace-flash")
    if real["model"]["variant"] != "real-flash":
        raise AssertionError("--real report does not contain real-flash")
    for name, report in (("multispace", multispace), ("real", real)):
        count = report["model"].get("trainable_parameters")
        if count != EXPECTED_PARAMETERS:
            raise AssertionError(
                f"{name} report has {count!r} parameters; expected {EXPECTED_PARAMETERS:,}"
            )
        if report.get("training_completion", {}).get("completed_schedule") is not True:
            raise AssertionError(f"{name} report does not prove completed pretraining")
        device_name = report.get("runtime", {}).get("device_name", "")
        if "A100" not in device_name.upper():
            raise AssertionError(f"{name} report was not produced on an A100: {device_name!r}")
        if report.get("runtime", {}).get("autocast_dtype") != "torch.bfloat16":
            raise AssertionError(f"{name} report did not use BF16 autocast")

    if multispace["protocol"] != real["protocol"]:
        raise AssertionError("paired reports do not have identical protocol manifests")
    fingerprint = multispace["protocol"]["dataset_sha256"]
    if not fingerprint or fingerprint != real["protocol"]["dataset_sha256"]:
        raise AssertionError("paired synthetic-dataset fingerprints differ")

    multispace_cells = multispace["cells"]
    real_cells = real["cells"]
    if set(multispace_cells) != set(real_cells):
        raise AssertionError("paired reports contain different MQAR cells")
    paired_accuracy_differences: list[float] = []
    paired_nll_differences: list[float] = []
    multispace_only_correct = 0
    real_only_correct = 0
    both_correct = 0
    both_incorrect = 0
    per_cell: dict[str, Mapping] = {}
    metadata_fields = (
        "context_length",
        "binding_count",
        "distractor_count",
        "query_distance",
        "query_distance_fraction",
        "difficulty",
        "examples",
    )
    for key in sorted(multispace_cells):
        multispace_cell = multispace_cells[key]
        real_cell = real_cells[key]
        for field in metadata_fields:
            if multispace_cell[field] != real_cell[field]:
                raise AssertionError(f"cell {key} differs on paired field {field}")
        multispace_correct = multispace_cell["example_correct"]
        real_correct = real_cell["example_correct"]
        multispace_nll = multispace_cell["example_nll"]
        real_nll = real_cell["example_nll"]
        paired_count = int(multispace_cell["examples"])
        if not (
            len(multispace_correct)
            == len(real_correct)
            == len(multispace_nll)
            == len(real_nll)
            == paired_count
        ):
            raise AssertionError(f"cell {key} has incomplete per-example outcomes")
        for ms_correct, baseline_correct, ms_nll, baseline_nll in zip(
            multispace_correct, real_correct, multispace_nll, real_nll
        ):
            paired_accuracy_differences.append(int(ms_correct) - int(baseline_correct))
            paired_nll_differences.append(float(ms_nll) - float(baseline_nll))
            if ms_correct and baseline_correct:
                both_correct += 1
            elif ms_correct:
                multispace_only_correct += 1
            elif baseline_correct:
                real_only_correct += 1
            else:
                both_incorrect += 1
        per_cell[key] = _summary_delta(multispace_cell, real_cell)

    overall = _summary_delta(
        multispace["summaries"]["overall"], real["summaries"]["overall"]
    )
    overall.update(
        {
            "paired_accuracy_difference_ci": _mean_confidence_interval(
                paired_accuracy_differences
            ),
            "paired_masked_nll_difference_ci": _mean_confidence_interval(
                paired_nll_differences
            ),
            "paired_outcomes": {
                "both_correct": both_correct,
                "multispace_only_correct": multispace_only_correct,
                "real_only_correct": real_only_correct,
                "both_incorrect": both_incorrect,
            },
        }
    )
    accuracy_ci = overall["paired_accuracy_difference_ci"]
    nll_ci = overall["paired_masked_nll_difference_ci"]
    if accuracy_ci["ci95_low"] > 0 and nll_ci["ci95_high"] < 0:
        conclusion = "multispace_better_on_accuracy_and_nll"
    elif accuracy_ci["ci95_high"] < 0 and nll_ci["ci95_low"] > 0:
        conclusion = "real_better_on_accuracy_and_nll"
    else:
        conclusion = "mixed_or_inconclusive"

    group_names = (
        "by_context_length",
        "by_binding_count",
        "by_distractor_count",
        "by_query_distance_fraction",
        "by_difficulty",
    )
    return {
        "benchmark": "masked-mqar-paired-comparison",
        "protocol": multispace["protocol"],
        "dataset_sha256": fingerprint,
        "parameter_matched": True,
        "trainable_parameters_each": EXPECTED_PARAMETERS,
        "conclusion": conclusion,
        "overall": overall,
        "groups": {
            name: _group_deltas(
                multispace["summaries"][name], real["summaries"][name]
            )
            for name in group_names
        },
        "cells": per_cell,
        "source_reports": {
            "multispace": multispace.get("model_path"),
            "real": real.get("model_path"),
        },
    }


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multispace", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_reports(_load_report(args.multispace), _load_report(args.real))
    atomic_write_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), **report["overall"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
