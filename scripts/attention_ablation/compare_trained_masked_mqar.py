#!/usr/bin/env python3
"""Compare paired trained-MQAR reports across transfer seeds."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import compare_masked_mqar as paired


PROTOCOL_VERSION = "trained-masked-mqar-v1"


def load_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"trained MQAR report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("benchmark") != "trained-masked-mqar":
        raise ValueError(f"not a trained masked-MQAR report: {path}")
    return report


def projection(report: Mapping) -> dict:
    """Project a trained test split into the strict paired comparator schema."""
    test = report["test"]
    return {
        "benchmark": "masked-mqar",
        "model_path": report["source_model_path"],
        "model": report["model"],
        "training_completion": report["transfer_completion"],
        "runtime": report["runtime"],
        "protocol": {
            "version": report["protocol_version"],
            "curriculum": report["curriculum"],
            "split_manifests": report["split_manifests"],
            "dataset_sha256": report["split_manifests"]["test"]["dataset_sha256"],
        },
        "cells": test["cells"],
        "summaries": test["summaries"],
    }


def mean_seed_metric(values: Sequence[float]) -> dict[str, float | int]:
    return paired._mean_confidence_interval(values)


def compare_seed(multispace: Mapping, real: Mapping, seed: int) -> dict:
    if multispace["curriculum"]["transfer_seed"] != seed:
        raise AssertionError("multispace transfer seed does not match path")
    if real["curriculum"]["transfer_seed"] != seed:
        raise AssertionError("real transfer seed does not match path")
    if multispace["curriculum"] != real["curriculum"]:
        raise AssertionError("paired optimizer/curriculum budgets differ")
    if multispace["split_manifests"] != real["split_manifests"]:
        raise AssertionError("paired generated train/validation/test data differ")
    fingerprints = {
        manifest["dataset_sha256"]
        for manifest in multispace["split_manifests"].values()
    }
    if len(fingerprints) != 3:
        raise AssertionError("split fingerprints are not disjoint")
    result = paired.compare_reports(projection(multispace), projection(real))
    result["transfer_seed"] = seed
    result["split_manifests"] = multispace["split_manifests"]
    result["training"] = {
        "multispace": multispace["training_metrics"],
        "real": real["training_metrics"],
    }
    return result


def compare_root(root: Path, seeds: Sequence[int]) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("transfer seeds must be non-empty and unique")
    seed_results = {}
    for seed in seeds:
        seed_root = root / f"seed-{seed}"
        multispace_path = seed_root / "multispace-flash" / "report.json"
        real_path = seed_root / "real-flash" / "report.json"
        seed_results[str(seed)] = compare_seed(
            load_report(multispace_path), load_report(real_path), seed
        )

    accuracy_deltas = [
        float(result["overall"]["accuracy_difference_multispace_minus_real"])
        for result in seed_results.values()
    ]
    nll_deltas = [
        float(result["overall"]["masked_nll_difference_multispace_minus_real"])
        for result in seed_results.values()
    ]
    throughput_ratios = [
        float(result["overall"]["real_over_multispace_throughput_ratio"])
        for result in seed_results.values()
    ]
    groups = {}
    for group_name in next(iter(seed_results.values()))["groups"]:
        groups[group_name] = {}
        group_keys = next(iter(seed_results.values()))["groups"][group_name]
        for group_key in group_keys:
            groups[group_name][group_key] = {
                "accuracy_delta_across_seeds": mean_seed_metric(
                    [
                        result["groups"][group_name][group_key][
                            "accuracy_difference_multispace_minus_real"
                        ]
                        for result in seed_results.values()
                    ]
                ),
                "masked_nll_delta_across_seeds": mean_seed_metric(
                    [
                        result["groups"][group_name][group_key][
                            "masked_nll_difference_multispace_minus_real"
                        ]
                        for result in seed_results.values()
                    ]
                ),
            }
    return {
        "benchmark": "trained-masked-mqar-multiseed-comparison",
        "protocol_version": PROTOCOL_VERSION,
        "transfer_seeds": list(seeds),
        "parameter_matched": True,
        "trainable_parameters_each": paired.EXPECTED_PARAMETERS,
        "aggregate": {
            "accuracy_delta_multispace_minus_real": mean_seed_metric(accuracy_deltas),
            "masked_nll_delta_multispace_minus_real": mean_seed_metric(nll_deltas),
            "real_over_multispace_throughput_ratio": mean_seed_metric(
                throughput_ratios
            ),
        },
        "groups": groups,
        "seeds": seed_results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
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
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_root(args.root, args.seeds)
    atomic_write_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), **report["aggregate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
