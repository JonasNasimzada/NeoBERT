#!/usr/bin/env python3
"""Prepare the pinned CIFAR-10 input used by the paired LRA adaptation.

This is data-only preprocessing and deliberately performs no model operation.
The 45k/5k split and RGB-to-grayscale conversion follow the official LRA
image input pipeline.  Outputs are plain NumPy arrays so every GPU run reads
exactly the same bytes without repeatedly decoding PNG payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset


DATASET_REPO = "uoft-cs/cifar10"
DATASET_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
SOURCE_FILES = {
    "train": (
        "plain_text/train-00000-of-00001.parquet",
        "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
    ),
    "test": (
        "plain_text/test-00000-of-00001.parquet",
        "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
    ),
}
EXPECTED_ROWS = {"train": 50_000, "test": 10_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def grayscale_lra(image: Any) -> np.ndarray:
    """Match tf.image.rgb_to_grayscale(uint8), used by official LRA.

    TensorFlow and torchvision use the same 0.2989/0.5870/0.1140 luma
    weights for this operation.  Conversion back to uint8 truncates toward
    zero.  CIFAR inputs are nonnegative, so astype(uint8) is exact here.
    """

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.shape != (32, 32, 3):
        raise RuntimeError(f"unexpected CIFAR-10 image shape: {rgb.shape}")
    work = rgb.astype(np.float32)
    gray = work[..., 0] * 0.2989 + work[..., 1] * 0.5870 + work[..., 2] * 0.1140
    return gray.astype(np.uint8).reshape(1024)


def write_split(dataset: Any, split: str, output_root: Path) -> dict[str, Any]:
    expected = EXPECTED_ROWS[split]
    if len(dataset) != expected:
        raise RuntimeError(f"{split} row count {len(dataset)} != {expected}")
    pixels_tmp = output_root / f"{split}_pixels.npy.tmp"
    labels_tmp = output_root / f"{split}_labels.npy.tmp"
    pixels_path = output_root / f"{split}_pixels.npy"
    labels_path = output_root / f"{split}_labels.npy"

    pixels = np.lib.format.open_memmap(
        pixels_tmp, mode="w+", dtype=np.uint8, shape=(expected, 1024)
    )
    labels = np.lib.format.open_memmap(
        labels_tmp, mode="w+", dtype=np.uint8, shape=(expected,)
    )
    for index, row in enumerate(dataset):
        pixels[index] = grayscale_lra(row["img"])
        label = int(row["label"])
        if not 0 <= label < 10:
            raise RuntimeError(f"invalid CIFAR-10 label at row {index}: {label}")
        labels[index] = label
        if (index + 1) % 5000 == 0:
            print(f"prepared {split}: {index + 1}/{expected}", flush=True)
    pixels.flush()
    labels.flush()
    del pixels, labels
    os.replace(pixels_tmp, pixels_path)
    os.replace(labels_tmp, labels_path)

    return {
        "rows": expected,
        "pixels": {
            "path": str(pixels_path.resolve()),
            "shape": [expected, 1024],
            "dtype": "uint8",
            "sha256": sha256(pixels_path),
        },
        "labels": {
            "path": str(labels_path.resolve()),
            "shape": [expected],
            "dtype": "uint8",
            "sha256": sha256(labels_path),
        },
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    source_manifest: dict[str, Any] = {}
    for split, (relative, expected_hash) in SOURCE_FILES.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected_hash:
            raise RuntimeError(
                f"source checksum mismatch for {path}: {observed} != {expected_hash}"
            )
        paths[split] = path
        source_manifest[split] = {
            "path": str(path),
            "sha256": observed,
            "hub_path": relative,
        }

    raw = load_dataset(
        "parquet",
        data_files={split: str(path) for split, path in paths.items()},
    )
    outputs = {
        split: write_split(raw[split], split, output_root)
        for split in ("train", "test")
    }
    manifest = {
        "schema_version": 1,
        "benchmark": "Long Range Arena",
        "task": "Image / sequential grayscale CIFAR-10",
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "source_files": source_manifest,
        },
        "preprocessing": {
            "official_reference": (
                "lra_benchmarks/image/input_pipeline.py@get_cifar10_datasets"
            ),
            "rgb_to_grayscale": "floor(0.2989*R + 0.5870*G + 0.1140*B)",
            "flatten_order": "row-major HWC after singleton channel removal",
            "sequence_length": 1024,
            "augmentation": "none",
            "train_validation_split": {
                "train": "source train[0:45000]",
                "validation": "source train[45000:50000]",
            },
            "test_split": "source test[0:10000]",
        },
        "outputs": outputs,
    }
    atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
