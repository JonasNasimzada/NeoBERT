#!/usr/bin/env python3
"""Fail the SuperGLUE gate unless every paired A100 preflight is fair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


TASKS = ("boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc")
EXPECTED_BASE_PARAMETERS = 99_985_152


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("SuperGLUE fairness gate must run on an A100")

    checks = []
    for task in TASKS:
        reports = []
        for variant in ("multispace", "real"):
            path = args.input_root / task / variant / f"seed-{args.seed}" / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "ok" or not report["protocol"]["preflight"]:
                raise RuntimeError(f"invalid preflight report: {path}")
            if report["model_metadata"]["base_parameters"] != EXPECTED_BASE_PARAMETERS:
                raise RuntimeError(f"parameter mismatch: {path}")
            if report["training"]["optimizer_steps"] != 1:
                raise RuntimeError(f"preflight did not execute one optimizer step: {path}")
            if not math.isfinite(float(report["primary_score"])):
                raise RuntimeError(f"non-finite preflight metric: {path}")
            reports.append(report)
        left, right = reports
        for field in ("task_head_initialization_sha256", "protocol_sha256"):
            if left[field] != right[field]:
                raise RuntimeError(f"unpaired {field} for task {task}")
        checks.append(
            {
                "task": task,
                "task_head_initialization_sha256": left[
                    "task_head_initialization_sha256"
                ],
                "protocol_sha256": left["protocol_sha256"],
            }
        )

    payload = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "exact_base_parameters": EXPECTED_BASE_PARAMETERS,
        "paired_tasks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
