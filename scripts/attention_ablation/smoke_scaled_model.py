#!/usr/bin/env python3
"""Run one disposable BF16 A100 training step for a scaled attention model.

This is a model/kernel validation harness, not a training entry point. It loads
one production YAML and permits only layer-count and sequence-length reductions
for staged smoke checks. The direct FlashAttention path deliberately receives
no padding mask or packed-document IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_IMPORT_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT.parent)
for import_root in reversed(LOCAL_IMPORT_ROOTS):
    import_root_string = str(import_root)
    if import_root_string not in sys.path:
        sys.path.insert(0, import_root_string)

from neobert.model import NeoBERTConfig, NeoBERTLMHead


VOCAB_SIZE = 30_522
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 103
PRODUCTION_SEQUENCE_LENGTH = 1_024
HIDDEN_SIZE = 768
NUM_ATTENTION_HEADS = 12
HEAD_DIMENSION = 64
HEADS_PER_SPACE = 4
SHARED_PARAMETERS = 23_441_664
PARAMETERS_PER_BLOCK = 8_504_832
SCALE_LAYERS = {"200m": 21, "300m": 33}
SCALE_PARAMETERS = {"200m": 202_043_136, "300m": 304_101_120}
VARIANT_INTERMEDIATE_SIZE = {"real": 4_000, "multispace": 2_464}


def config_path(size: str, variant: str) -> Path:
    return (
        PROJECT_ROOT
        / "conf"
        / "model"
        / f"attention-ablation-{variant}-{size}.yaml"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=tuple(SCALE_LAYERS))
    parser.add_argument(
        "--variant",
        required=True,
        choices=tuple(VARIANT_INTERMEDIATE_SIZE),
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="smoke-only depth override (default: full production depth)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=PRODUCTION_SEQUENCE_LENGTH,
        help="smoke sequence length (default: 1024)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_030)
    parser.add_argument(
        "--require-a100",
        action="store_true",
        help="reject CUDA devices other than NVIDIA A100 (SM80)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser.parse_args()


def load_smoke_config(
    size: str,
    variant: str,
    *,
    layers: int | None,
    sequence_length: int,
) -> tuple[NeoBERTConfig, Path, int]:
    production_layers = SCALE_LAYERS[size]
    smoke_layers = production_layers if layers is None else layers
    if not 1 <= smoke_layers <= production_layers:
        raise ValueError(
            f"--layers must be in [1, {production_layers}], got {smoke_layers}"
        )
    if not 2 <= sequence_length <= PRODUCTION_SEQUENCE_LENGTH:
        raise ValueError(
            "--sequence-length must be in "
            f"[2, {PRODUCTION_SEQUENCE_LENGTH}], got {sequence_length}"
        )

    path = config_path(size, variant)
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict):
        raise TypeError(f"{path} must contain a mapping")

    expected_space = "real" if variant == "real" else "multispace"
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": production_layers,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "intermediate_size": VARIANT_INTERMEDIATE_SIZE[variant],
        "rope": True,
        "rms_norm": True,
        "embedding_rms_norm": False,
        "hidden_act": "gelu",
        "fused_swiglu": False,
        "dropout": 0,
        "attention_dropout": 0,
        "tie_word_embeddings": True,
        "lm_head_bias": False,
        "ngpt": False,
        "attention_space": expected_space,
        "attention_backend": "flash",
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise AssertionError(
                f"{path} declares {key}={values.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    if variant == "multispace" and values.get("multispace_cuda_streams") is not True:
        raise AssertionError(f"{path} must enable multispace_cuda_streams")

    if "attention_spaces" in values:
        schedule = list(values["attention_spaces"])
        if schedule != [expected_space] * production_layers:
            raise AssertionError(f"unexpected production space schedule: {schedule!r}")
        values["attention_spaces"] = schedule[:smoke_layers]
    if "attention_backends" in values:
        schedule = list(values["attention_backends"])
        if schedule != ["flash"] * production_layers:
            raise AssertionError(f"unexpected production backend schedule: {schedule!r}")
        values["attention_backends"] = schedule[:smoke_layers]

    values.update(
        num_hidden_layers=smoke_layers,
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_TOKEN_ID,
        max_length=PRODUCTION_SEQUENCE_LENGTH,
    )
    config = NeoBERTConfig(**values)
    if config.attention_spaces != [expected_space] * smoke_layers:
        raise AssertionError(f"smoke attention-space schedule is invalid")
    if config.attention_backends != ["flash"] * smoke_layers:
        raise AssertionError(f"smoke attention-backend schedule is invalid")
    if config.dim_head != HEAD_DIMENSION:
        raise AssertionError(f"expected head dimension {HEAD_DIMENSION}")
    return config, path, smoke_layers


def require_cuda(*, require_a100: bool) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this smoke must run on a GPU")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if require_a100 and ("A100" not in name.upper() or capability != (8, 0)):
        raise RuntimeError(f"expected an NVIDIA A100 (SM80), found {name} {capability}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{name} does not report CUDA BF16 support")
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "capability": [capability[0], capability[1]],
        "total_memory_gib": round(properties.total_memory / 2**30, 3),
    }


def make_mlm_batch(
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    input_ids = torch.randint(
        104,
        VOCAB_SIZE,
        (batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    labels = torch.full_like(input_ids, -100)
    selected = (
        (torch.arange(sequence_length, device=device) % 8 == 0)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    labels[selected] = input_ids[selected]
    input_ids[selected] = MASK_TOKEN_ID
    return input_ids, labels, int(selected.sum().item())


def unique_parameter_count(model: torch.nn.Module) -> int:
    parameters = {id(parameter): parameter for parameter in model.parameters()}
    return sum(parameter.numel() for parameter in parameters.values())


def _assert_nonzero_finite_gradient(name: str, gradient: torch.Tensor | None) -> float:
    if gradient is None:
        raise RuntimeError(f"{name} has no gradient")
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError(f"{name} has a non-finite gradient")
    norm = float(torch.linalg.vector_norm(gradient.float()).detach())
    if norm == 0.0:
        raise RuntimeError(f"{name} gradient is identically zero")
    return norm


def attention_gradient_norms(
    model: NeoBERTLMHead,
    variant: str,
) -> tuple[dict[str, float], torch.nn.Parameter]:
    first_block = model.model.transformer_encoder[0]
    if variant == "real":
        return (
            {
                "real.qkv": _assert_nonzero_finite_gradient(
                    "real.qkv", first_block.qkv.weight.grad
                ),
                "real.out": _assert_nonzero_finite_gradient(
                    "real.out", first_block.wo.weight.grad
                ),
            },
            first_block.qkv.weight,
        )

    attention = first_block.complex_attention
    qkv_gradient = attention.qkv.weight.grad
    output_gradient = attention.out_proj.weight.grad
    if qkv_gradient is None or output_gradient is None:
        raise RuntimeError("multispace attention projections have missing gradients")
    qkv_rows_per_space = qkv_gradient.shape[0] // 3
    output_columns_per_space = output_gradient.shape[1] // 3
    norms: dict[str, float] = {}
    for index, space in enumerate(("complex", "split", "dual")):
        qkv_slice = qkv_gradient[
            index * qkv_rows_per_space : (index + 1) * qkv_rows_per_space
        ]
        output_slice = output_gradient[
            :, index * output_columns_per_space : (index + 1) * output_columns_per_space
        ]
        norms[f"{space}.qkv"] = _assert_nonzero_finite_gradient(
            f"{space}.qkv", qkv_slice
        )
        norms[f"{space}.out"] = _assert_nonzero_finite_gradient(
            f"{space}.out", output_slice
        )
    return norms, attention.qkv.weight


def main() -> int:
    args = parse_args()
    device_info = require_cuda(require_a100=args.require_a100)
    config, production_path, smoke_layers = load_smoke_config(
        args.size,
        args.variant,
        layers=args.layers,
        sequence_length=args.sequence_length,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    # Parameters stay FP32; CUDA autocast supplies BF16 QKV to the strict
    # FlashAttention paths used by real, complex, split, and dual attention.
    model = NeoBERTLMHead(config).to(device=device).train()
    actual_parameters = unique_parameter_count(model)
    expected_parameters = SHARED_PARAMETERS + smoke_layers * PARAMETERS_PER_BLOCK
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"instantiated {actual_parameters:,} parameters, expected "
            f"{expected_parameters:,}"
        )
    if smoke_layers == SCALE_LAYERS[args.size]:
        if actual_parameters != SCALE_PARAMETERS[args.size]:
            raise RuntimeError("full production model does not match its size contract")

    first_block = model.model.transformer_encoder[0]
    if args.variant == "multispace":
        attention = first_block.complex_attention
        if tuple(attention.space_names) != ("complex", "split", "dual"):
            raise RuntimeError("multispace layer has an unexpected group schedule")
        if attention.heads_per_space != HEADS_PER_SPACE:
            raise RuntimeError("multispace layer does not have four heads per space")
        if attention.out_proj.in_features != 2 * HIDDEN_SIZE:
            raise RuntimeError("multispace layer discarded a component before mixing")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
    )
    input_ids, labels, target_count = make_mlm_batch(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Direct FlashAttention cannot represent arbitrary packed-document or
        # padding masks, so this full, padding-free synthetic sequence passes
        # input IDs and MLM labels only.
        outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
    if outputs.loss is None or not bool(torch.isfinite(outputs.loss)):
        raise RuntimeError("MLM smoke loss is missing or non-finite")
    if outputs.logits.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 logits, got {outputs.logits.dtype}")
    loss_value = float(outputs.loss.detach())
    logits_dtype = str(outputs.logits.dtype).removeprefix("torch.")
    outputs.loss.backward()

    named_parameters = tuple(model.named_parameters())
    missing_gradients = [
        name
        for name, parameter in named_parameters
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        preview = ", ".join(missing_gradients[:8])
        raise RuntimeError(f"trainable parameters without gradients: {preview}")
    total_grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float("inf"),
        error_if_nonfinite=True,
        foreach=False,
    )
    gradient_norms, tracked_parameter = attention_gradient_norms(model, args.variant)

    tracked_before = tracked_parameter.detach().clone()
    del outputs
    optimizer.step()
    if torch.equal(tracked_before, tracked_parameter.detach()):
        raise RuntimeError("AdamW did not update the first attention QKV projection")
    if not bool(torch.isfinite(tracked_parameter).all()):
        raise RuntimeError("AdamW produced a non-finite QKV parameter")
    if not optimizer.state:
        raise RuntimeError("AdamW did not initialize optimizer state")
    del tracked_before
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_device_memory = torch.cuda.get_device_properties(0).total_memory
    optimizer_state_dtypes = sorted(
        {
            str(value.dtype).removeprefix("torch.")
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
    )

    result = {
        "schema_version": 1,
        "kind": "scaled_attention_gpu_smoke",
        "status": "passed",
        "size": args.size,
        "variant": args.variant,
        "production_config": str(production_path),
        "production_layers": SCALE_LAYERS[args.size],
        "smoke_layers": smoke_layers,
        "production_sequence_length": PRODUCTION_SEQUENCE_LENGTH,
        "smoke_sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "mlm_targets": target_count,
        "attention_space": config.attention_space,
        "attention_backend": config.attention_backend,
        "attention_heads": config.num_attention_heads,
        "heads_by_space": (
            {"real": NUM_ATTENTION_HEADS}
            if args.variant == "real"
            else {"complex": 4, "split": 4, "dual": 4}
        ),
        "head_dimension": config.dim_head,
        "intermediate_size": config.intermediate_size,
        "parameters": actual_parameters,
        "expected_full_production_parameters": SCALE_PARAMETERS[args.size],
        "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "autocast_dtype": "bfloat16",
        "logits_dtype": logits_dtype,
        "loss": loss_value,
        "gradient_norm": float(total_grad_norm.detach()),
        "first_layer_attention_gradient_norms": gradient_norms,
        "all_trainable_parameters_received_gradients": True,
        "adamw_step_completed": True,
        "optimizer_state_dtypes": optimizer_state_dtypes,
        "direct_flash_masking": "padding-free; no mask or document_ids",
        "device": device_info,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "peak_allocated_gib": round(peak_allocated / 2**30, 3),
        "peak_reserved_gib": round(peak_reserved / 2**30, 3),
        "peak_reserved_fraction": round(peak_reserved / total_device_memory, 4),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
