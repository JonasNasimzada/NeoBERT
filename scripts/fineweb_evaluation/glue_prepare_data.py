#!/usr/bin/env python3
"""Download and verify the immutable GLUE parquet evaluation snapshot.

The files are taken directly from the pinned ``nyu-mll/glue`` Hub commit.
Only public train/validation data and the unlabelled AX diagnostic split are
needed: GLUE test labels are private, so local scores must use validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASET_REPOSITORY = "nyu-mll/glue"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
BASE_URL = (
    "https://huggingface.co/datasets/nyu-mll/glue/resolve/"
    f"{DATASET_REVISION}"
)

# SHA-256 values are the Git-LFS object identifiers published by the pinned
# Hub tree.  Keeping them in source makes a moving cache or revision visible.
FILES: dict[str, tuple[int, str]] = {
    "ax/test-00000-of-00001.parquet": (
        80_767,
        "a07b802fe2d4968a1f7ccce9406826dc77e0d1dc53fea9491664bd8ebba8571a",
    ),
    "cola/train-00000-of-00001.parquet": (
        251_124,
        "2e7538afa2000e63f5343f16a758d75c452661a384208399d2035cd2fce45c33",
    ),
    "cola/validation-00000-of-00001.parquet": (
        37_551,
        "c14b7219a7d9f9fe3dd291fd000f6623ee413805eb108c9c49578ed50873e4ba",
    ),
    "mnli/train-00000-of-00001.parquet": (
        52_224_361,
        "49a4a5508b89b8fed2c6e81d2c47d00f4759050a7048c6cc5d95d31122ced3c1",
    ),
    "mnli/validation_matched-00000-of-00001.parquet": (
        1_214_936,
        "7f918c09d9c35446b8e8f06a5672f8ab704e2897fecbf52e2e154141f3d7c421",
    ),
    "mnli/validation_mismatched-00000-of-00001.parquet": (
        1_251_152,
        "04aba92823a954be36fe1b69b61eed334c9eb1009daba0dd79f69d77b87c535c",
    ),
    "mrpc/train-00000-of-00001.parquet": (
        649_281,
        "61fd41301e0e244b0420c4350a170c8e7cf64740335fc875a4af2d79af0df0af",
    ),
    "mrpc/validation-00000-of-00001.parquet": (
        75_678,
        "33c007dbf5bfa8463d87a13e6226df8c0fcf2596c2cd39d0f3bb79754e00f50f",
    ),
    "qnli/train-00000-of-00001.parquet": (
        17_528_917,
        "ebc7cb70a5bbde0b0336c3d51f31bb4df4673e908e8874b090b52169b1365c6c",
    ),
    "qnli/validation-00000-of-00001.parquet": (
        872_062,
        "e69311b81dc65589286091d9905a27617a90436dd215c7a59832fa8f4f336169",
    ),
    "qqp/train-00000-of-00001.parquet": (
        33_558_839,
        "4d6f02e643f7c36e9a4f7d4971a5ee9bd74063a319452fe6c87850c739774cd7",
    ),
    "qqp/validation-00000-of-00001.parquet": (
        3_729_274,
        "efd86a539c412d74874ee451573d7bd142f56c47fe36de033b9f367d8bb0fa71",
    ),
    "rte/train-00000-of-00001.parquet": (
        583_976,
        "a6252ab17015d718f6de1effe0980f7b158df63e3d16207cd8bd396b608e5147",
    ),
    "rte/validation-00000-of-00001.parquet": (
        69_020,
        "fb2aa2e04f551133ba663617a15ae133dc22b0f6a969bc0629b5ea6003ee9cf8",
    ),
    "sst2/train-00000-of-00001.parquet": (
        3_110_468,
        "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229",
    ),
    "sst2/validation-00000-of-00001.parquet": (
        72_819,
        "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
    ),
    "stsb/train-00000-of-00001.parquet": (
        502_065,
        "bbd93bbb988fd18437e02185fe3b2bd9a18350376c392e7820de9df1b247ed1f",
    ),
    "stsb/validation-00000-of-00001.parquet": (
        150_622,
        "152de7cf1fa34ee4df1c243bd209b02ade21a1d5c4fb3b7da5240f78e4000aa9",
    ),
    "wnli/train-00000-of-00001.parquet": (
        38_835,
        "40f4c0c60db68addeda8e9cbe25e6344cd99d5bbb80125535994a9a3141ee0a9",
    ),
    "wnli/validation-00000-of-00001.parquet": (
        11_067,
        "880037e45e03df868d5799ca21dc03f3a6378f0adf3c01c7bfc46b94fa61f1cb",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != size:
        raise RuntimeError(
            f"size mismatch for {path}: expected {size}, found {actual_size}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.download-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"temporary download already exists: {temporary}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NeoBERT-GLUE-evaluation/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download missing files.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for relative_path, (size, expected_sha256) in sorted(FILES.items()):
        destination = args.output / relative_path
        if not destination.exists():
            if args.verify_only:
                raise FileNotFoundError(destination)
            download(f"{BASE_URL}/{relative_path}", destination)
        verify(destination, size, expected_sha256)
        records.append(
            {
                "path": relative_path,
                "bytes": size,
                "sha256": expected_sha256,
                "url": f"{BASE_URL}/{relative_path}",
            }
        )

    payload = {
        "schema_version": 1,
        "benchmark": "GLUE",
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "dataset_card": "https://huggingface.co/datasets/nyu-mll/glue",
        "official_benchmark": "https://gluebenchmark.com/",
        "scoring_scope": (
            "public validation splits; GLUE test labels are private"
        ),
        "ax_scope": (
            "unlabelled diagnostic prediction export; excluded from aggregate"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    manifest_path = args.output / "data_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
