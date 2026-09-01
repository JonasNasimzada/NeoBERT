#!/usr/bin/env python3
"""Validate the paired GLUE A100 preflight and publish its manifest."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(f"preflight is not successful: {path}")
    if "A100" not in payload.get("device", "").upper():
        raise RuntimeError(f"preflight did not use an A100: {path}")
    if payload.get("base_parameters") != 99_985_152:
        raise RuntimeError(f"wrong encoder parameter count: {path}")
    if payload.get("finetune_attention_backend") != "torch":
        raise RuntimeError(f"wrong supervised attention backend: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multispace", required=True, type=Path)
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    multispace = load(args.multispace)
    real = load(args.real)
    if multispace.get("attention_space") != "multispace":
        raise RuntimeError("multispace preflight loaded the wrong architecture")
    if real.get("attention_space") != "real":
        raise RuntimeError("real preflight loaded the wrong architecture")

    paired_fields = (
        "head_initialization_sha256",
        "tokenizer_sha256",
        "data_manifest_sha256",
        "hyperparameters_sha256",
        "task",
        "seed",
        "base_parameters",
        "head_parameters",
        "total_trainable_parameters",
    )
    for field in paired_fields:
        if multispace.get(field) != real.get(field):
            raise RuntimeError(
                f"paired preflight mismatch for {field}: "
                f"{multispace.get(field)!r} != {real.get(field)!r}"
            )

    payload = {
        "schema_version": 1,
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairing_checks": {field: multispace[field] for field in paired_fields},
        "variants": {
            "multispace-flash": multispace,
            "real-flash": real,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
