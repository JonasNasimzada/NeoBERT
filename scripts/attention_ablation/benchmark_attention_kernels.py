#!/usr/bin/env python3
"""Reproduce the FlashAttention paper microbenchmark grids for this project.

The default grids follow FlashAttention Appendix E.6 (arXiv:2205.14135v2)
and FlashAttention-2 Section 4.1 (arXiv:2307.08691v1).  This is an attention
kernel benchmark: Q, K, and V are random tensors and projection layers are
deliberately excluded.

CUDA allocator peaks reported here describe live/reserved PyTorch allocations.
They are not measurements of HBM traffic or memory bandwidth.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor


FA1_SEQUENCE_LENGTHS = tuple(2**power for power in range(7, 17))
FA2_SEQUENCE_LENGTHS = (512, 1_024, 2_048, 4_096, 8_192, 16_384)
DEFAULT_VARIANTS = (
    "complex-native",
    "complex-torch",
    "complex-flash",
    "split-native",
    "split-torch",
    "real-torch",
    "real-flash",
    "split-flash",
    "dual-native",
    "dual-torch",
    "dual-flash",
)
ALL_VARIANTS = DEFAULT_VARIANTS

PROTOCOL_METADATA = {
    "fa1-e6": {
        "paper": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "arxiv": "2205.14135v2",
        "location": "Appendix E.6",
        "default_batch_size": 16,
        "default_heads": 8,
        "default_head_dim": 64,
        "default_sequence_lengths": list(FA1_SEQUENCE_LENGTHS),
        "default_dropout_probabilities": [0.0, 0.1],
        "default_padding_mask_values": [False, True],
        "causal": False,
        "padding_mask_protocol": (
            "deterministic right padding; each sample's valid length is drawn "
            "uniformly and inclusively from [sequence_length-20, sequence_length]"
        ),
        "default_repetitions": 100,
    },
    "fa2-4.1": {
        "paper": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        "arxiv": "2307.08691v1",
        "location": "Section 4.1",
        "default_total_tokens": 16_384,
        "default_hidden_size": 2_048,
        "default_head_dims": [64, 128],
        "default_sequence_lengths": list(FA2_SEQUENCE_LENGTHS),
        "default_dropout_probability": 0.0,
        "default_causal_mask_values": [False, True],
        "default_repetitions": 30,
    },
}

MEMORY_MEASUREMENT_NOTE = (
    "CUDA allocator allocated/reserved peaks for one forward followed by one "
    "backward pass. Baselines include Q/K/V, output-gradient tensors, and an "
    "optional padding mask. Incremental values are peak minus baseline. These "
    "are not HBM traffic or memory-bandwidth measurements."
)
FLOP_MEASUREMENT_NOTE = (
    "paper_nominal_* uses the FlashAttention paper convention: forward = "
    "4*batch*sequence^2*heads*head_dim (halved for causal attention), backward "
    "= 2.5*forward, and combined = 3.5*forward. logical_* multiplies that "
    "nominal count by the represented algebra's main-attention work estimate; "
    "only paper_nominal_* is directly comparable with the paper plots."
)

RESUME_DEVICE_FIELDS = (
    "torch_version",
    "torch_cuda_version",
    "cuda_device_name",
    "cuda_capability",
    "cuda_total_memory_bytes",
    "cuda_multiprocessor_count",
)


@dataclass(frozen=True)
class VariantSpec:
    algebra: str
    backend: str
    physical_components: int
    logical_algebra_multiplier: float


VARIANT_SPECS = {
    "complex-native": VariantSpec("ordinary_complex", "native", 2, 2.0),
    "complex-torch": VariantSpec("ordinary_complex", "torch", 2, 2.0),
    "complex-flash": VariantSpec("ordinary_complex", "flash", 2, 2.0),
    "split-native": VariantSpec("split_complex", "native", 2, 2.0),
    "split-torch": VariantSpec("split_complex", "torch", 2, 2.0),
    "split-flash": VariantSpec("split_complex", "flash", 2, 2.0),
    "real-torch": VariantSpec("real", "torch", 1, 1.0),
    "real-flash": VariantSpec("real", "flash", 1, 1.0),
    "dual-native": VariantSpec("dual_number", "native", 2, 3.0),
    "dual-torch": VariantSpec("dual_number", "torch", 2, 3.0),
    "dual-flash": VariantSpec("dual_number", "flash", 2, 3.0),
}
BACKEND_TARGETS = {
    "complex-native": "pytorch-sdpa-math-packed-complex",
    "complex-torch": "pytorch-sdpa-auto-packed-complex",
    "complex-flash": "pytorch-sdpa-flash-packed-complex",
    "split-native": "custom-aten-split-complex",
    "split-torch": "pytorch-sdpa-auto-two-split-channels",
    "split-flash": "pytorch-sdpa-flash-one-packed-split-complex-call",
    "real-torch": "pytorch-sdpa-auto",
    "real-flash": "pytorch-sdpa-flash",
    "dual-native": "custom-aten-dual-number",
    "dual-torch": "pytorch-jvp-sdpa",
    "dual-flash": "triton-fused-dual-flash",
}


@dataclass(frozen=True)
class BenchmarkCase:
    protocol: str
    variant: str
    batch_size: int
    heads: int
    sequence_length: int
    head_dim: int
    causal: bool
    padding_mask: bool
    dropout_p: float
    repetitions: int
    warmup_repetitions: int
    dtype: str = "float16"

    @property
    def hidden_size(self) -> int:
        return self.heads * self.head_dim

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.sequence_length

    @property
    def case_id(self) -> str:
        dropout = format(self.dropout_p, "g").replace(".", "p")
        if self.causal:
            mask = "causal"
        elif self.padding_mask:
            mask = "padding"
        else:
            mask = "unmasked"
        return (
            f"{self.protocol}-{self.variant}-b{self.batch_size}-h{self.heads}-"
            f"s{self.sequence_length}-d{self.head_dim}-{mask}-dropout{dropout}"
        )


class UnsupportedCase(RuntimeError):
    """A requested row cannot faithfully execute the named backend."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _parse_float_csv(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not 0.0 <= item < 1.0 for item in result):
        raise argparse.ArgumentTypeError("dropout values must be in [0, 1)")
    return result


def _parse_bool_csv(value: str) -> tuple[bool, ...]:
    aliases = {
        "0": False,
        "false": False,
        "no": False,
        "1": True,
        "true": True,
        "yes": True,
    }
    result: list[bool] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized not in aliases:
            raise argparse.ArgumentTypeError(
                "boolean lists accept true/false, yes/no, or 1/0"
            )
        result.append(aliases[normalized])
    if not result:
        raise argparse.ArgumentTypeError("expected at least one boolean")
    return tuple(result)


def _canonical_variants(values: Sequence[str]) -> tuple[str, ...]:
    variants: list[str] = []
    for value in values:
        for item in value.split(","):
            variant = item.strip().lower().replace("_", "-")
            if not variant:
                continue
            if variant not in VARIANT_SPECS:
                raise ValueError(
                    f"unknown variant {variant!r}; choices: {', '.join(ALL_VARIANTS)}"
                )
            if variant not in variants:
                variants.append(variant)
    if not variants:
        raise ValueError("at least one variant is required")
    return tuple(variants)


def paper_nominal_flops(case: BenchmarkCase) -> dict[str, float]:
    """Return the nominal FLOP convention used by the FA2 benchmark script."""
    forward = (
        4
        * case.batch_size
        * case.sequence_length**2
        * case.heads
        * case.head_dim
    )
    if case.causal:
        forward //= 2
    forward = float(forward)
    return {
        "forward": forward,
        "backward": 2.5 * forward,
        "combined": 3.5 * forward,
    }


def estimated_input_bytes(case: BenchmarkCase, *, element_size: int) -> int:
    spec = VARIANT_SPECS[case.variant]
    elements = (
        3
        * spec.physical_components
        * case.batch_size
        * case.heads
        * case.sequence_length
        * case.head_dim
    )
    return elements * element_size


def generate_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    variants = _canonical_variants(args.variants)
    cases: list[BenchmarkCase] = []
    if args.protocol in ("both", "fa1"):
        for variant in variants:
            for sequence_length in args.fa1_sequence_lengths:
                for dropout_p in args.fa1_dropouts:
                    for padding_mask in args.fa1_padding_mask_values:
                        cases.append(
                            BenchmarkCase(
                                protocol="fa1-e6",
                                variant=variant,
                                batch_size=args.fa1_batch_size,
                                heads=args.fa1_heads,
                                sequence_length=sequence_length,
                                head_dim=args.fa1_head_dim,
                                causal=False,
                                padding_mask=padding_mask,
                                dropout_p=dropout_p,
                                repetitions=args.fa1_repetitions,
                                warmup_repetitions=args.warmup_repetitions,
                                dtype=args.dtype,
                            )
                        )
    if args.protocol in ("both", "fa2"):
        for sequence_length in args.fa2_sequence_lengths:
            if args.fa2_total_tokens % sequence_length:
                raise ValueError(
                    "FA2 total tokens must be divisible by every sequence length; "
                    f"got {args.fa2_total_tokens} and {sequence_length}"
                )
        for head_dim in args.fa2_head_dims:
            if args.fa2_hidden_size % head_dim:
                raise ValueError(
                    "FA2 hidden size must be divisible by every head dimension; "
                    f"got {args.fa2_hidden_size} and {head_dim}"
                )
        for variant in variants:
            for head_dim in args.fa2_head_dims:
                heads = args.fa2_hidden_size // head_dim
                for sequence_length in args.fa2_sequence_lengths:
                    batch_size = args.fa2_total_tokens // sequence_length
                    for causal in args.fa2_causal_values:
                        cases.append(
                            BenchmarkCase(
                                protocol="fa2-4.1",
                                variant=variant,
                                batch_size=batch_size,
                                heads=heads,
                                sequence_length=sequence_length,
                                head_dim=head_dim,
                                causal=causal,
                                padding_mask=False,
                                dropout_p=0.0,
                                repetitions=args.fa2_repetitions,
                                warmup_repetitions=args.warmup_repetitions,
                                dtype=args.dtype,
                            )
                        )
    return cases[: args.max_rows] if args.max_rows is not None else cases


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {name}") from error
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dtype must be float16, bfloat16, or float32")
    return dtype


def _flatten_tensors(value: Any) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        tensors: list[Tensor] = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tuple(tensors)
    raise TypeError(f"attention output contains non-tensor value {type(value).__name__}")


def _make_inputs(
    case: BenchmarkCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[Any, Any, Any]:
    shape = (
        case.batch_size,
        case.heads,
        case.sequence_length,
        case.head_dim,
    )
    generator = torch.Generator(device=device).manual_seed(seed)

    def component() -> Tensor:
        return torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )

    components = VARIANT_SPECS[case.variant].physical_components
    values: list[Any] = []
    for _ in range(3):
        values.append(component() if components == 1 else (component(), component()))
    return values[0], values[1], values[2]


def make_padding_mask(
    case: BenchmarkCase,
    *,
    device: torch.device,
    seed: int,
) -> Tensor | None:
    """Construct the deterministic right-padding mask used by FA1 Appendix E.6."""
    if not case.padding_mask:
        return None
    minimum_valid = max(1, case.sequence_length - 20)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    valid_lengths = torch.randint(
        minimum_valid,
        case.sequence_length + 1,
        (case.batch_size,),
        generator=generator,
        device="cpu",
    )
    positions = torch.arange(case.sequence_length, device="cpu")
    return (positions.unsqueeze(0) >= valid_lengths.unsqueeze(1)).to(device=device)


def _attention_callable(case: BenchmarkCase) -> Callable[[Any, Any, Any, Tensor | None], Any]:
    """Resolve implementations lazily so importing this module needs no extension."""
    spec = VARIANT_SPECS[case.variant]
    scale = case.head_dim**-0.5
    if spec.backend in ("flash", "flash_fused") and case.padding_mask:
        raise UnsupportedCase(
            "strict FlashAttention does not accept key-padding masks; the row is "
            "reported as unsupported instead of falling back to another backend"
        )
    if (
        spec.backend in ("flash", "flash_fused")
        and case.dropout_p
        and spec.algebra in ("split_complex", "dual_number")
    ):
        raise UnsupportedCase(
            f"{spec.algebra} FlashAttention cannot preserve one shared dropout "
            "sample across both components; this row is explicitly unsupported"
        )

    if spec.algebra == "real":
        try:
            from complex_attention import efficient_attention
        except (ImportError, OSError) as error:
            raise UnsupportedCase(
                "real benchmark requires the ComplexAttention attention_backends package"
            ) from error

        def run_real(
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Tensor | None,
        ) -> Tensor:
            return efficient_attention(
                query,
                key,
                value,
                is_causal=case.causal,
                scale=scale,
                dropout_p=case.dropout_p,
                backend=spec.backend,
                key_padding_mask=key_padding_mask,
            )

        return run_real

    if spec.algebra == "ordinary_complex":
        try:
            from complex_attention import complex_dot_product_attention as complex_attention
        except (ImportError, OSError) as error:
            raise UnsupportedCase(
                "ordinary-complex benchmark requires complex_attention"
            ) from error

        def run_complex(
            query: Any,
            key: Any,
            value: Any,
            key_padding_mask: Tensor | None,
        ) -> Any:
            output, _ = complex_attention(
                query,
                key,
                value,
                is_causal=case.causal,
                scale=scale,
                dropout_p=case.dropout_p,
                backend=spec.backend,
                key_padding_mask=key_padding_mask,
            )
            return output

        return run_complex

    if spec.algebra == "split_complex":
        if spec.backend == "native" and case.dropout_p:
            raise UnsupportedCase(
                "split-native uses a Torch fallback when dropout is nonzero; "
                "the row is omitted to avoid labeling fallback time as native"
            )
        try:
            from complex_attention import split_complex_attention
        except (ImportError, OSError) as error:
            raise UnsupportedCase("split-complex benchmark requires complex_attention") from error

        def run_split(
            query: Any,
            key: Any,
            value: Any,
            key_padding_mask: Tensor | None,
        ) -> Any:
            output, _ = split_complex_attention(
                query,
                key,
                value,
                is_causal=case.causal,
                scale=scale,
                dropout_p=case.dropout_p,
                backend=spec.backend,
                key_padding_mask=key_padding_mask,
            )
            return output

        return run_split

    if spec.algebra == "dual_number":
        if spec.backend == "native" and case.dropout_p:
            raise UnsupportedCase(
                "dual-native uses a Torch JVP fallback when dropout is nonzero; "
                "the row is omitted to avoid labeling fallback time as native"
            )
        try:
            import complex_attention

            dual_attention = getattr(complex_attention, "dual_attention")
        except (ImportError, OSError, AttributeError) as error:
            raise UnsupportedCase(
                "dual variants require an installation exporting "
                "complex_attention.dual_attention"
            ) from error

        def run_dual(
            query: Any,
            key: Any,
            value: Any,
            key_padding_mask: Tensor | None,
        ) -> Any:
            output, _ = dual_attention(
                query,
                key,
                value,
                is_causal=case.causal,
                scale=scale,
                dropout_p=case.dropout_p,
                backend=spec.backend,
                key_padding_mask=key_padding_mask,
            )
            return output

        return run_dual

    raise AssertionError(f"unhandled algebra {spec.algebra}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _backward_once(
    output: Any,
    inputs: tuple[Tensor, ...],
    grad_outputs: tuple[Tensor, ...],
    *,
    retain_graph: bool,
) -> None:
    outputs = _flatten_tensors(output)
    if len(outputs) != len(grad_outputs):
        raise RuntimeError(
            f"expected {len(grad_outputs)} output components, got {len(outputs)}"
        )
    gradients = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    del gradients


def _warm_up(
    operation: Callable[[], Any],
    inputs: tuple[Tensor, ...],
    grad_outputs: tuple[Tensor, ...],
    *,
    repetitions: int,
    device: torch.device,
) -> None:
    for _ in range(repetitions):
        output = operation()
        _backward_once(output, inputs, grad_outputs, retain_graph=False)
        del output
    _synchronize(device)


def _time_cuda_forward(
    operation: Callable[[], Any],
    *,
    repetitions: int,
    device: torch.device,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(repetitions):
        output = operation()
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end) / repetitions
    del output, start, end
    return elapsed_ms


def _time_cuda_backward(
    operation: Callable[[], Any],
    inputs: tuple[Tensor, ...],
    grad_outputs: tuple[Tensor, ...],
    *,
    repetitions: int,
) -> float:
    output = operation()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        _backward_once(output, inputs, grad_outputs, retain_graph=True)
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end) / repetitions
    del output, start, end
    return elapsed_ms


def _measure_cuda_memory(
    operation: Callable[[], Any],
    inputs: tuple[Tensor, ...],
    grad_outputs: tuple[Tensor, ...],
    *,
    device: torch.device,
) -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    output = operation()
    _backward_once(output, inputs, grad_outputs, retain_graph=False)
    _synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del output
    return {
        "cuda_baseline_allocated_bytes": baseline_allocated,
        "cuda_baseline_reserved_bytes": baseline_reserved,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_incremental_peak_allocated_bytes": max(
            0, peak_allocated - baseline_allocated
        ),
        "cuda_incremental_peak_reserved_bytes": max(0, peak_reserved - baseline_reserved),
    }


def _failure_status(error: BaseException) -> str:
    message = str(error).lower()
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "oom"
    if isinstance(error, (UnsupportedCase, ImportError, AttributeError, NotImplementedError)):
        return "unsupported"
    unsupported_fragments = (
        "not supported",
        "unsupported",
        "requires cuda",
        "requires triton",
        "requires compute capability",
        "requires the custom",
        "no available kernel",
        "no viable backend",
        "not implemented",
        "does not support",
    )
    if any(fragment in message for fragment in unsupported_fragments):
        return "unsupported"
    return "error"


def _base_row(case: BenchmarkCase, *, element_size: int) -> dict[str, Any]:
    spec = VARIANT_SPECS[case.variant]
    nominal = paper_nominal_flops(case)
    row: dict[str, Any] = {
        **asdict(case),
        "case_id": case.case_id,
        "status": "pending",
        "algebra": spec.algebra,
        "backend_requested": spec.backend,
        "backend_target": BACKEND_TARGETS[case.variant],
        "backend_effective": None,
        "physical_components": spec.physical_components,
        "logical_algebra_multiplier": spec.logical_algebra_multiplier,
        "hidden_size": case.hidden_size,
        "total_tokens": case.total_tokens,
        "masked": case.causal or case.padding_mask,
        "mask_kind": (
            "causal" if case.causal else "key_padding" if case.padding_mask else "none"
        ),
        "input_qkv_bytes": estimated_input_bytes(case, element_size=element_size),
        "padding_mask_bytes": (
            case.batch_size * case.sequence_length if case.padding_mask else 0
        ),
        "grad_output_bytes": (
            spec.physical_components
            * case.batch_size
            * case.heads
            * case.sequence_length
            * case.head_dim
            * element_size
        ),
    }
    for phase, flops in nominal.items():
        row[f"paper_nominal_{phase}_flops"] = flops
        row[f"logical_{phase}_flops"] = flops * spec.logical_algebra_multiplier
    return row


def _throughput_tflops(flops: float, milliseconds: float) -> float:
    return flops / (milliseconds / 1_000.0) / 1.0e12


def _tokens_per_second(total_tokens: int, milliseconds: float) -> float:
    return total_tokens / (milliseconds / 1_000.0)


def benchmark_case(
    case: BenchmarkCase,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    dtype = _torch_dtype(case.dtype)
    row = _base_row(case, element_size=torch.empty((), dtype=dtype).element_size())
    started = time.perf_counter()
    query = key = value = key_padding_mask = None
    try:
        if device.type != "cuda":
            raise UnsupportedCase("timing and CUDA memory metrics require a CUDA device")
        operation_impl = _attention_callable(case)
        row["backend_effective"] = BACKEND_TARGETS[case.variant]
        query, key, value = _make_inputs(case, device=device, dtype=dtype, seed=seed)
        key_padding_mask = make_padding_mask(case, device=device, seed=seed)
        if key_padding_mask is not None:
            valid_lengths = key_padding_mask.logical_not().sum(dim=1)
            row.update(
                padding_valid_length_min=int(valid_lengths.min().item()),
                padding_valid_length_max=int(valid_lengths.max().item()),
                padding_valid_length_mean=float(valid_lengths.float().mean().item()),
            )
        inputs = _flatten_tensors((query, key, value))
        grad_outputs = tuple(
            torch.randn_like(component)
            for component in _flatten_tensors(query)
        )
        operation = lambda: operation_impl(query, key, value, key_padding_mask)

        _warm_up(
            operation,
            inputs,
            grad_outputs,
            repetitions=case.warmup_repetitions,
            device=device,
        )
        row.update(
            _measure_cuda_memory(
                operation,
                inputs,
                grad_outputs,
                device=device,
            )
        )
        forward_ms = _time_cuda_forward(
            operation,
            repetitions=case.repetitions,
            device=device,
        )
        backward_ms = _time_cuda_backward(
            operation,
            inputs,
            grad_outputs,
            repetitions=case.repetitions,
        )
        combined_ms = forward_ms + backward_ms
        row.update(
            status="ok",
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            combined_ms=combined_ms,
            forward_tokens_per_second=_tokens_per_second(
                case.total_tokens, forward_ms
            ),
            backward_tokens_per_second=_tokens_per_second(
                case.total_tokens, backward_ms
            ),
            combined_tokens_per_second=_tokens_per_second(
                case.total_tokens, combined_ms
            ),
            paper_nominal_forward_tflops=_throughput_tflops(
                row["paper_nominal_forward_flops"], forward_ms
            ),
            paper_nominal_backward_tflops=_throughput_tflops(
                row["paper_nominal_backward_flops"], backward_ms
            ),
            paper_nominal_combined_tflops=_throughput_tflops(
                row["paper_nominal_combined_flops"], combined_ms
            ),
            logical_forward_tflops=_throughput_tflops(
                row["logical_forward_flops"], forward_ms
            ),
            logical_backward_tflops=_throughput_tflops(
                row["logical_backward_flops"], backward_ms
            ),
            logical_combined_tflops=_throughput_tflops(
                row["logical_combined_flops"], combined_ms
            ),
        )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        row.update(
            status=_failure_status(error),
            error_type=type(error).__name__,
            error_message=str(error),
        )
    finally:
        row["row_wall_time_seconds"] = time.perf_counter() - started
        del query, key, value, key_padding_mask
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested_device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        resolved = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(resolved)
        metadata.update(
            cuda_device_index=resolved,
            cuda_device_name=properties.name,
            cuda_capability=f"{properties.major}.{properties.minor}",
            cuda_total_memory_bytes=properties.total_memory,
            cuda_multiprocessor_count=properties.multi_processor_count,
        )
        uuid = getattr(properties, "uuid", None)
        if uuid is not None:
            metadata["cuda_device_uuid"] = str(uuid)
    for name in (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_JOB_NODELIST",
    ):
        if value := os.environ.get(name):
            metadata[name.lower()] = value
    return metadata


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _resume_signature(
    args: argparse.Namespace,
    cases: Sequence[BenchmarkCase],
) -> dict[str, Any]:
    """Describe every input that can change rows or their W&B destination."""
    return {
        "version": 1,
        "seed": args.seed,
        "device": str(args.device),
        "fail_on_error": bool(args.fail_on_error),
        "cases": [asdict(case) for case in cases],
        "wandb": {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "name": args.wandb_name,
            "id": args.wandb_id,
            "tags": list(args.wandb_tags),
        },
    }


def _legacy_resume_signature(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct a signature for schema-v1 files written before --resume."""
    cli = payload.get("cli")
    if not isinstance(cli, Mapping):
        raise ValueError("resume file does not contain a valid cli configuration")
    try:
        legacy_args = argparse.Namespace(**dict(cli))
        legacy_cases = generate_cases(legacy_args)
        return _resume_signature(legacy_args, legacy_cases)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "resume file cli configuration cannot reconstruct its benchmark cases"
        ) from error


def _validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    cases: Sequence[BenchmarkCase],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a mutable compatible running or complete payload."""
    if payload.get("schema_version") != 1:
        raise ValueError(
            "resume file has an unsupported schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("benchmark") != "attention-kernels":
        raise ValueError("resume file is not an attention-kernels benchmark")
    status = payload.get("status")
    if status not in ("running", "complete"):
        raise ValueError(
            f"resume file status must be 'running' or 'complete', got {status!r}"
        )

    expected_signature = _resume_signature(args, cases)
    stored_signature = payload.get("resume_signature")
    if stored_signature is None:
        stored_signature = _legacy_resume_signature(payload)
    if stored_signature != expected_signature:
        raise ValueError(
            "resume file is incompatible with the requested cases, seed, device, "
            "error policy, or W&B destination"
        )

    stored_device = payload.get("device")
    if not isinstance(stored_device, Mapping):
        raise ValueError("resume file does not contain valid device metadata")
    mismatched_device_fields = [
        name
        for name in RESUME_DEVICE_FIELDS
        if name in stored_device
        and name in metadata
        and stored_device[name] != metadata[name]
    ]
    if mismatched_device_fields:
        raise ValueError(
            "resume device/software is incompatible in: "
            + ", ".join(mismatched_device_fields)
        )

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("resume file rows must be a list")
    if len(rows) > len(cases):
        raise ValueError(
            f"resume file has {len(rows)} rows but this run has only {len(cases)} cases"
        )
    completed_statuses = {"ok", "unsupported", "oom", "error"}
    case_fields = tuple(BenchmarkCase.__dataclass_fields__)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"resume row {index} is not an object")
        case = cases[index]
        mismatched_case_fields = [
            name for name in case_fields if row.get(name) != getattr(case, name)
        ]
        if row.get("case_id") != case.case_id:
            mismatched_case_fields.append("case_id")
        if mismatched_case_fields:
            raise ValueError(
                f"resume row {index} is not the expected case prefix; mismatched: "
                + ", ".join(mismatched_case_fields)
            )
        if row.get("status") not in completed_statuses:
            raise ValueError(
                f"resume row {index} is not complete: status={row.get('status')!r}"
            )
    if status == "complete" and len(rows) != len(cases):
        raise ValueError(
            f"complete resume file has {len(rows)} of {len(cases)} expected rows"
        )
    return dict(payload)


def _read_resume_payload(
    output: Path,
    *,
    args: argparse.Namespace,
    cases: Sequence[BenchmarkCase],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        loaded = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read resume file {output}: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"resume file {output} must contain a JSON object")
    return _validate_resume_payload(
        loaded,
        args=args,
        cases=cases,
        metadata=metadata,
    )


def _wandb_init(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    cases: Sequence[BenchmarkCase],
):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is required unless --wandb-mode=disabled is selected"
        ) from error
    config = {
        "benchmark": "attention-kernels",
        "protocol": args.protocol,
        "variants": list(_canonical_variants(args.variants)),
        "case_count": len(cases),
        "dtype": args.dtype,
        "warmup_repetitions": args.warmup_repetitions,
        "paper_protocols": PROTOCOL_METADATA,
        "device": dict(metadata),
        "memory_measurement_note": MEMORY_MEASUREMENT_NOTE,
        "flop_measurement_note": FLOP_MEASUREMENT_NOTE,
    }
    options: dict[str, Any] = {
        "project": args.wandb_project,
        "mode": args.wandb_mode,
        "config": config,
        "job_type": "attention-kernel-benchmark",
        "name": args.wandb_name,
        "group": args.wandb_group,
        "entity": args.wandb_entity,
        "tags": args.wandb_tags,
    }
    if args.wandb_id:
        options.update(id=args.wandb_id, resume="allow")
    return wandb.init(**{key: value for key, value in options.items() if value is not None})


def _wandb_log_row(run: Any, row: Mapping[str, Any], index: int) -> None:
    if run is None:
        return
    status_codes = {"ok": 0, "unsupported": 1, "oom": 2, "error": 3}
    payload: dict[str, Any] = {
        "kernel_benchmark/row_index": index,
        "kernel_benchmark/status_code": status_codes.get(str(row["status"]), 4),
        "kernel_benchmark/batch_size": row["batch_size"],
        "kernel_benchmark/heads": row["heads"],
        "kernel_benchmark/sequence_length": row["sequence_length"],
        "kernel_benchmark/head_dim": row["head_dim"],
        "kernel_benchmark/causal": int(bool(row["causal"])),
        "kernel_benchmark/padding_mask": int(bool(row["padding_mask"])),
        "kernel_benchmark/dropout_p": row["dropout_p"],
        "kernel_benchmark/input_qkv_bytes": row["input_qkv_bytes"],
        "kernel_benchmark/padding_mask_bytes": row["padding_mask_bytes"],
        "kernel_benchmark/logical_algebra_multiplier": row[
            "logical_algebra_multiplier"
        ],
    }
    metric_names = (
        "forward_ms",
        "backward_ms",
        "combined_ms",
        "forward_tokens_per_second",
        "backward_tokens_per_second",
        "combined_tokens_per_second",
        "paper_nominal_forward_tflops",
        "paper_nominal_backward_tflops",
        "paper_nominal_combined_tflops",
        "logical_forward_tflops",
        "logical_backward_tflops",
        "logical_combined_tflops",
        "cuda_baseline_allocated_bytes",
        "cuda_baseline_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "cuda_incremental_peak_allocated_bytes",
        "cuda_incremental_peak_reserved_bytes",
        "row_wall_time_seconds",
    )
    payload.update(
        {f"kernel_benchmark/{name}": row[name] for name in metric_names if name in row}
    )
    run.log(payload, step=index)


def _wandb_log_results(run: Any, rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    if run is None:
        return
    import wandb

    columns = sorted({key for row in rows for key in row})
    table = wandb.Table(
        columns=columns,
        data=[[row.get(column) for column in columns] for row in rows],
    )
    run.log({"kernel_benchmark/results": table})
    artifact_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{run.name}-kernel-benchmark")
    artifact = wandb.Artifact(artifact_name, type="attention-kernel-benchmark")
    artifact.add_file(str(output.resolve()))
    run.log_artifact(artifact)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("both", "fa1", "fa2"), default="both")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Paper-exact default is float16; overrides are intended for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-repetitions", type=int, default=10)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", type=Path, default=Path("attention-kernels.json"))
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a compatible running output file from its completed row "
            "prefix; a compatible complete file is left unchanged."
        ),
    )

    parser.add_argument("--fa1-sequence-lengths", type=_parse_int_csv, default=FA1_SEQUENCE_LENGTHS)
    parser.add_argument("--fa1-batch-size", type=int, default=16)
    parser.add_argument("--fa1-heads", type=int, default=8)
    parser.add_argument("--fa1-head-dim", type=int, default=64)
    parser.add_argument("--fa1-dropouts", type=_parse_float_csv, default=(0.0, 0.1))
    parser.add_argument(
        "--fa1-padding-mask-values",
        type=_parse_bool_csv,
        default=(False, True),
    )
    parser.add_argument("--fa1-repetitions", type=int, default=100)

    parser.add_argument("--fa2-sequence-lengths", type=_parse_int_csv, default=FA2_SEQUENCE_LENGTHS)
    parser.add_argument("--fa2-total-tokens", type=int, default=16_384)
    parser.add_argument("--fa2-hidden-size", type=int, default=2_048)
    parser.add_argument("--fa2-head-dims", type=_parse_int_csv, default=(64, 128))
    parser.add_argument("--fa2-causal-values", type=_parse_bool_csv, default=(False, True))
    parser.add_argument("--fa2-repetitions", type=int, default=30)

    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "complex-attention-ablation"),
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument(
        "--wandb-tags",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=["attention-kernel-benchmark", "flashattention-paper-protocols"],
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "warmup_repetitions",
        "fa1_batch_size",
        "fa1_heads",
        "fa1_head_dim",
        "fa1_repetitions",
        "fa2_total_tokens",
        "fa2_hidden_size",
        "fa2_repetitions",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    cases = generate_cases(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for paper timing and memory measurements")
    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    metadata = device_metadata(device)
    output = args.output.expanduser().resolve()
    if args.resume and output.exists():
        payload = _read_resume_payload(
            output,
            args=args,
            cases=cases,
            metadata=metadata,
        )
        if payload["status"] == "complete":
            error_count = sum(row["status"] == "error" for row in payload["rows"])
            print(
                f"Benchmark already complete: {len(cases)}/{len(cases)} rows in {output}",
                flush=True,
            )
            return int(error_count > 0)
        resumed_at = _utc_now()
        payload["resume_signature"] = _resume_signature(args, cases)
        payload.setdefault("resume_history", []).append(
            {
                "resumed_at": resumed_at,
                "completed_rows": len(payload["rows"]),
                "remaining_rows": len(cases) - len(payload["rows"]),
                "device": metadata,
            }
        )
        payload["updated_at"] = resumed_at
        atomic_write_json(output, payload)
        print(
            f"Resuming benchmark at row {len(payload['rows']) + 1}/{len(cases)} "
            f"from {output}",
            flush=True,
        )
    else:
        payload = {
            "schema_version": 1,
            "benchmark": "attention-kernels",
            "status": "running",
            "started_at": _utc_now(),
            "paper_protocols": PROTOCOL_METADATA,
            "memory_measurement_note": MEMORY_MEASUREMENT_NOTE,
            "flop_measurement_note": FLOP_MEASUREMENT_NOTE,
            "device": metadata,
            "cli": vars(args) | {"output": str(output)},
            "resume_signature": _resume_signature(args, cases),
            "rows": [],
        }
        atomic_write_json(output, payload)
    run = _wandb_init(args, metadata, cases)
    try:
        first_pending_index = len(payload["rows"])
        for index in range(first_pending_index, len(cases)):
            case = cases[index]
            row = benchmark_case(case, device=device, seed=args.seed + index)
            payload["rows"].append(row)
            payload["updated_at"] = _utc_now()
            atomic_write_json(output, payload)
            _wandb_log_row(run, row, index)
            print(
                f"[{index + 1}/{len(cases)}] {case.case_id}: {row['status']}",
                flush=True,
            )
            if args.fail_on_error and row["status"] != "ok":
                raise RuntimeError(
                    f"benchmark row failed ({row['status']}): {case.case_id}: "
                    f"{row.get('error_message', '')}"
                )
        counts = {
            status: sum(row["status"] == status for row in payload["rows"])
            for status in ("ok", "unsupported", "oom", "error")
        }
        payload.update(
            status="complete",
            completed_at=_utc_now(),
            status_counts=counts,
        )
        atomic_write_json(output, payload)
        _wandb_log_results(run, payload["rows"], output)
        if run is not None:
            for status, count in counts.items():
                run.summary[f"kernel_benchmark/{status}_rows"] = count
            run.summary["kernel_benchmark/output"] = str(output)
    finally:
        if run is not None:
            run.finish()
    return int(counts["error"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
