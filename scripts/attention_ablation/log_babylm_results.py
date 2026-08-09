#!/usr/bin/env python3
"""Parse an official BabyLM results tree and log every numeric result to W&B."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from numbers import Real
from pathlib import Path


NAMESPACE = "benchmark/babylm"
TEXT_SUFFIXES = {
    "",
    ".log",
    ".md",
    ".out",
    ".report",
    ".result",
    ".results",
    ".txt",
}
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KEY_VALUE_PATTERN = re.compile(
    rf"^\s*([^:#][^:]*?)\s*:\s*({NUMBER_PATTERN})\s*(%)?\s*$"
)
SECTION_HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
BARE_NUMBER_PATTERN = re.compile(rf"^\s*({NUMBER_PATTERN})\s*(%)?\s*$")


def _segment(value: str) -> str:
    segment = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value.strip().lower(),
    ).strip("_.-")
    return segment or "value"


def _path_segments(path: Path) -> tuple[str, ...]:
    without_suffix = path.with_suffix("") if path.suffix else path
    return tuple(_segment(part) for part in without_suffix.parts)


def flatten_numeric_values(
    value,
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], float]]:
    """Recursively yield finite JSON numbers while excluding booleans."""
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from flatten_numeric_values(
                value[key],
                (*prefix, _segment(str(key))),
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from flatten_numeric_values(item, (*prefix, str(index)))
        return
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            yield prefix or ("value",), numeric


def parse_key_value_report(text: str) -> dict[tuple[str, ...], float]:
    metrics: dict[tuple[str, ...], float] = {}
    pending_section: tuple[str, ...] | None = None
    for line in text.splitlines():
        header_match = SECTION_HEADER_PATTERN.match(line)
        if header_match is not None:
            pending_section = (_segment(header_match.group(1)),)
            continue

        match = KEY_VALUE_PATTERN.match(line)
        if match is not None:
            key, raw_value, percent = match.groups()
            value = float(raw_value)
            if percent:
                value /= 100.0
            segments = tuple(
                _segment(part)
                for part in re.split(r"\s*/\s*", key)
                if part.strip()
            ) or ("value",)
            if segments in metrics:
                raise ValueError(f"duplicate report key: {key!r}")
            metrics[segments] = value
            pending_section = None
            continue

        if pending_section is not None:
            bare_match = BARE_NUMBER_PATTERN.match(line)
            if bare_match is not None:
                raw_value, percent = bare_match.groups()
                value = float(raw_value)
                if percent:
                    value /= 100.0
                if pending_section in metrics:
                    raise ValueError(
                        f"duplicate report section: {pending_section[0]!r}"
                    )
                metrics[pending_section] = value
                pending_section = None
            elif line.strip():
                pending_section = None
    return metrics


def collect_metrics(
    results_root: Path,
    *,
    exclude: Path | None = None,
    max_file_bytes: int = 10 * 1024 * 1024,
) -> tuple[dict[str, float], dict[str, str]]:
    """Collect namespaced metrics and their source files from a result tree."""
    root = results_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"results root is not a directory: {root}")
    excluded = exclude.resolve() if exclude is not None else None
    metrics: dict[str, float] = {}
    sources: dict[str, str] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or (excluded is not None and path.resolve() == excluded):
            continue
        if path.stat().st_size > max_file_bytes:
            continue
        relative = path.relative_to(root)
        file_prefix = _path_segments(relative)

        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot parse JSON result {path}: {error}") from error
            parsed = {}
            for value_path, value in flatten_numeric_values(payload):
                if value_path in parsed:
                    raise ValueError(
                        f"normalized JSON metric collision in {path}: {value_path}"
                    )
                parsed[value_path] = value
        elif path.suffix.lower() in TEXT_SUFFIXES:
            try:
                parsed = parse_key_value_report(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
        else:
            continue

        for value_path, value in parsed.items():
            metric = "/".join((NAMESPACE, *file_prefix, *value_path))
            if metric in metrics:
                raise ValueError(
                    f"metric collision for {metric!r}: "
                    f"{sources[metric]} and {relative.as_posix()}"
                )
            metrics[metric] = value
            sources[metric] = relative.as_posix()

    return metrics, sources


def _atomic_write_json(path: Path, payload: Mapping) -> None:
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


def _identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return identifier[:120] or "babylm-results"


def _log_to_wandb(metrics: Mapping[str, float], output: Path, args) -> None:
    if args.mode == "disabled":
        return
    import wandb

    id_prefix = getattr(args, "id_prefix", "").strip()
    run_id = "-".join(
        part
        for part in (
            id_prefix,
            f"{args.variant}-seed-{args.seed}-babylm",
        )
        if part
    )
    run = wandb.init(
        id=_identifier(run_id),
        resume="allow",
        project=args.project,
        entity=args.entity or None,
        name=args.name or f"{args.variant}-babylm",
        group=args.group or args.variant,
        job_type="benchmark",
        mode=args.mode,
        tags=["benchmark", "babylm", args.variant],
        config={
            "variant": args.variant,
            "experiment_id": id_prefix or None,
            "seed": args.seed,
            "results_root": str(args.results.resolve()),
            "metric_count": len(metrics),
        },
    )
    try:
        if metrics:
            run.log(dict(metrics))
        artifact = wandb.Artifact(
            name=_identifier(f"{args.variant}-babylm-results-{run.id}"),
            type="benchmark-results",
            metadata={"variant": args.variant, "benchmark": "babylm"},
        )
        artifact.add_dir(str(args.results.resolve()), name="official_results")
        artifact.add_file(str(output.resolve()), name="parsed_metrics.json")
        run.log_artifact(artifact)
    finally:
        run.finish()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Official BabyLM results root")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--id-prefix",
        default="",
        help="Experiment prefix used to isolate the resumable W&B run id",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--project", default="complex-attention-ablation")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--group", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--mode",
        "--wandb-mode",
        dest="mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--max-file-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--allow-empty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.variant = args.variant.strip().lower().replace("_", "-")
    if args.max_file_bytes <= 0:
        raise ValueError("--max-file-bytes must be positive")
    if not args.results.is_dir():
        raise NotADirectoryError(f"results root is not a directory: {args.results}")
    if args.output is None:
        args.output = args.results.resolve().parent / (
            f"{args.results.name}.{_identifier(args.variant)}.parsed.json"
        )

    metrics, sources = collect_metrics(
        args.results,
        exclude=args.output,
        max_file_bytes=args.max_file_bytes,
    )
    if not metrics and not args.allow_empty:
        raise RuntimeError(f"no numeric BabyLM metrics found under {args.results}")

    report = {
        "schema_version": 1,
        "benchmark": "babylm",
        "variant": args.variant,
        "seed": args.seed,
        "results_root": str(args.results.resolve()),
        "metric_count": len(metrics),
        "metrics": metrics,
        "sources": sources,
    }
    _atomic_write_json(args.output, report)
    _log_to_wandb(metrics, args.output, args)
    print(f"Parsed {len(metrics):,} metrics from {args.results.resolve()}")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
