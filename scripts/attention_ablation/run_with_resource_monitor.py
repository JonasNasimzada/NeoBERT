#!/usr/bin/env python3
"""Run a benchmark command while sampling descendant CPU/GPU memory usage."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence


def descendant_pids(root_pid: int) -> set[int]:
    """Return a best-effort snapshot of a Linux process tree."""
    discovered = {int(root_pid)}
    pending = [int(root_pid)]
    while pending:
        pid = pending.pop()
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = children_path.read_text(encoding="utf-8").split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for value in children:
            child = int(value)
            if child not in discovered:
                discovered.add(child)
                pending.append(child)
    return discovered


def process_tree_rss_bytes(pids: set[int]) -> int:
    total_kib = 0
    for pid in pids:
        try:
            lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2:
                    total_kib += int(fields[1])
                break
    return total_kib * 1024


def compute_process_gpu_bytes(pids: set[int]) -> int:
    """Sum nvidia-smi's process memory for the sampled process tree."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    total_mib = 0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            used_mib = int(fields[1])
        except ValueError:
            continue
        if pid in pids:
            total_mib += used_mib
    return total_mib * 1024 * 1024


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


def monitor_command(
    command: Sequence[str],
    *,
    output: Path,
    sample_interval: float,
) -> int:
    if not command:
        raise ValueError("a command is required after --")
    if sample_interval <= 0:
        raise ValueError("sample interval must be positive")

    started = time.monotonic()
    process = subprocess.Popen(list(command), start_new_session=True)
    peak_host_rss_bytes = 0
    peak_gpu_process_memory_bytes = 0
    samples = 0
    gpu_samples = 0
    gpu_sampling_error: str | None = None

    def forward_signal(signum, _frame) -> None:
        if process.poll() is None:
            os.killpg(process.pid, signum)

    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, forward_signal)

    try:
        while True:
            pids = descendant_pids(process.pid)
            peak_host_rss_bytes = max(
                peak_host_rss_bytes,
                process_tree_rss_bytes(pids),
            )
            try:
                gpu_bytes = compute_process_gpu_bytes(pids)
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as error:
                if gpu_sampling_error is None:
                    gpu_sampling_error = f"{type(error).__name__}: {error}"
            else:
                peak_gpu_process_memory_bytes = max(
                    peak_gpu_process_memory_bytes,
                    gpu_bytes,
                )
                gpu_samples += 1
            samples += 1
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(sample_interval)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if process.poll() is None:
            process.wait()

    elapsed_seconds = time.monotonic() - started
    report = {
        "schema_version": 1,
        "monitor": "process_tree_nvidia_smi",
        "command_executable": command[0],
        "exit_code": int(process.returncode),
        "elapsed_seconds": elapsed_seconds,
        "sample_interval_seconds": sample_interval,
        "samples": samples,
        "gpu_samples": gpu_samples,
        "peak_host_rss_bytes": peak_host_rss_bytes,
        "peak_gpu_process_memory_bytes": peak_gpu_process_memory_bytes,
        "gpu_memory_source": "nvidia-smi compute-process sampling",
        "gpu_sampling_error": gpu_sampling_error,
    }
    _atomic_write_json(output, report)
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    return monitor_command(
        command,
        output=args.output,
        sample_interval=args.sample_interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
