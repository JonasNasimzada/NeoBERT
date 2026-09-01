#!/usr/bin/env python3
"""Run one disposable BF16 CUDA optimizer step for an OptiBERTneo variant.

This is deliberately a kernel/model validation harness, not a training entry
point.  It loads one of the two production model YAML files, then applies only
the requested layer-count and sequence-length reductions for the smoke run.
No dataset is read and no checkpoint is written.
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


PRODUCTION_SEQUENCE_LENGTH = 1_024
PRODUCTION_LAYERS = 28
VOCAB_SIZE = 50_265
PAD_TOKEN_ID = 1
MASK_TOKEN_ID = 50_264
MODEL_CONFIGS = {
    "real": PROJECT_ROOT / "conf" / "model" / "optibertneo-198m.yaml",
    "multispace": (
        PROJECT_ROOT
        / "conf"
        / "model"
        / "optibertneo-198m-multispace.yaml"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a test-only CUDA BF16 forward/backward/AdamW step using an "
            "OptiBERTneo production model YAML."
        )
    )
    parser.add_argument("--variant", required=True, choices=tuple(MODEL_CONFIGS))
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="smoke-only layer override (default: 1; production: 28)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="smoke-only sequence override (default: 128; production: 1024)",
    )
    parser.add_argument("--seed", type=int, default=19_804)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="synthetic per-device microbatch (default: 1)",
    )
    parser.add_argument(
        "--require-a100",
        action="store_true",
        help="reject any CUDA device other than an NVIDIA A100 (SM80)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="also exercise the training pipeline's torch.compile wrapper",
    )
    return parser.parse_args()


def load_smoke_config(
    variant: str,
    *,
    layers: int,
    sequence_length: int,
) -> NeoBERTConfig:
    if not 1 <= layers <= PRODUCTION_LAYERS:
        raise ValueError(
            f"--layers must be in [1, {PRODUCTION_LAYERS}], got {layers}"
        )
    if not 2 <= sequence_length <= PRODUCTION_SEQUENCE_LENGTH:
        raise ValueError(
            "--sequence-length must be in "
            f"[2, {PRODUCTION_SEQUENCE_LENGTH}], got {sequence_length}"
        )

    config_path = MODEL_CONFIGS[variant]
    values = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(values, dict):
        raise TypeError(f"{config_path} must contain a mapping")

    expected_space = "real" if variant == "real" else "multispace"
    production_contract = {
        "hidden_size": 768,
        "num_hidden_layers": PRODUCTION_LAYERS,
        "num_attention_heads": 12,
        "attention_space": expected_space,
        "attention_backend": "flex",
        "attention_dropout": 0,
        "tie_word_embeddings": True,
        "lm_head_bias": False,
    }
    for key, expected in production_contract.items():
        if values.get(key) != expected:
            raise AssertionError(
                f"{config_path} declares {key}={values.get(key)!r}, "
                f"expected {expected!r}"
            )

    # Explicit schedules are not currently present in these YAMLs.  If they
    # are added later, retain only the homogeneous production prefix selected
    # by this smoke-only layer override.
    if "attention_spaces" in values:
        spaces = list(values["attention_spaces"])
        if spaces != [expected_space] * PRODUCTION_LAYERS:
            raise AssertionError(f"unexpected production space schedule: {spaces!r}")
        values["attention_spaces"] = spaces[:layers]
    if "attention_backends" in values:
        backends = list(values["attention_backends"])
        if backends != ["flex"] * PRODUCTION_LAYERS:
            raise AssertionError(
                f"unexpected production backend schedule: {backends!r}"
            )
        values["attention_backends"] = backends[:layers]

    values.update(
        num_hidden_layers=layers,
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_TOKEN_ID,
        max_length=PRODUCTION_SEQUENCE_LENGTH,
    )
    config = NeoBERTConfig(**values)
    if config.attention_spaces != [expected_space] * layers:
        raise AssertionError(
            f"smoke space schedule is {config.attention_spaces!r}"
        )
    if config.attention_backends != ["flex"] * layers:
        raise AssertionError(
            f"smoke backend schedule is {config.attention_backends!r}"
        )
    return config


def require_cuda(*, require_a100: bool) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this smoke must run on a GPU")

    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if require_a100 and "A100" not in name.upper():
        raise RuntimeError(f"expected an NVIDIA A100, found {name}")
    if require_a100 and capability != (8, 0):
        raise RuntimeError(f"expected SM80, found compute capability {capability}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{name} does not report CUDA BF16 support")

    return {
        "name": name,
        "capability": f"{capability[0]}.{capability[1]}",
        "total_memory_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 2**30,
            3,
        ),
    }


def make_packed_mlm_batch(
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    # The midpoint creates two nonempty documents in every packed sequence.
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    input_ids = torch.randint(
        2,
        MASK_TOKEN_ID,
        (batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    midpoint = sequence_length // 2
    document_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    document_ids[:, midpoint:] = 1
    if torch.unique(document_ids).numel() < 2:
        raise AssertionError("packed smoke batch must contain at least two documents")

    labels = torch.full_like(input_ids, -100)
    selected = (
        (torch.arange(sequence_length, device=device) % 8 == 0)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )
    # Guarantee that each document contributes at least one MLM target.
    selected[:, 0] = True
    selected[:, midpoint] = True
    labels[selected] = input_ids[selected]
    input_ids[selected] = MASK_TOKEN_ID
    return input_ids, labels, document_ids, int(selected.sum().item())


def unique_parameter_count(model: torch.nn.Module) -> int:
    parameters = {id(parameter): parameter for parameter in model.parameters()}
    return sum(parameter.numel() for parameter in parameters.values())


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device_info = require_cuda(require_a100=args.require_a100)
    config = load_smoke_config(
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

    # Parameters remain FP32 while CUDA autocast runs the supported operations
    # and activations in BF16, matching the mixed-precision training contract.
    uncompiled_model = NeoBERTLMHead(config).to(device=device).train()
    model = (
        torch.compile(uncompiled_model, fullgraph=False)
        if args.compile
        else uncompiled_model
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
    )
    input_ids, labels, document_ids, target_count = make_packed_mlm_batch(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            document_ids=document_ids,
            labels=labels,
            return_dict=True,
        )
    if outputs.loss is None:
        raise RuntimeError("MLM smoke did not return a loss")
    if outputs.logits.dtype != torch.bfloat16:
        raise RuntimeError(
            f"expected BF16 autocast logits, got {outputs.logits.dtype}"
        )
    if not bool(torch.isfinite(outputs.loss)):
        raise RuntimeError("smoke loss is not finite")

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
    )
    if not bool(torch.isfinite(total_grad_norm)):
        raise RuntimeError("aggregate gradient norm is not finite")

    first_block = uncompiled_model.model.transformer_encoder[0]
    tracked_parameter = (
        first_block.qkv.weight
        if args.variant == "real"
        else first_block.complex_attention.qkv.weight
    )
    if tracked_parameter.grad is None:
        raise RuntimeError("first attention QKV projection has no gradient")
    if not bool(torch.isfinite(tracked_parameter.grad).all()):
        raise RuntimeError("first attention QKV projection has a non-finite gradient")
    if int(torch.count_nonzero(tracked_parameter.grad).item()) == 0:
        raise RuntimeError("first attention QKV projection gradient is identically zero")

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
    current_allocated = torch.cuda.memory_allocated(device)
    optimizer_state_dtypes = sorted(
        {
            str(value.dtype).removeprefix("torch.")
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
    )

    result = {
        "kind": "disposable_optibertneo_gpu_smoke",
        "variant": args.variant,
        "production_config": str(MODEL_CONFIGS[args.variant]),
        "production_layers": PRODUCTION_LAYERS,
        "smoke_layers": args.layers,
        "production_sequence_length": PRODUCTION_SEQUENCE_LENGTH,
        "smoke_sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "packed_documents": 2 * args.batch_size,
        "mlm_targets": target_count,
        "attention_space": config.attention_space,
        "attention_backend": config.attention_backend,
        "attention_heads": config.num_attention_heads,
        "head_dimension": config.dim_head,
        "unique_parameters": unique_parameter_count(model),
        "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "autocast_dtype": "bfloat16",
        "logits_dtype": logits_dtype,
        "loss": loss_value,
        "gradient_norm": float(total_grad_norm.detach()),
        "adamw_step_completed": True,
        "torch_compile": args.compile,
        "optimizer_state_dtypes": optimizer_state_dtypes,
        "device": device_info,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "peak_allocated_gib": round(peak_allocated / 2**30, 3),
        "peak_reserved_gib": round(peak_reserved / 2**30, 3),
        "current_allocated_gib": round(current_allocated / 2**30, 3),
    }
    if args.variant == "multispace":
        result["known_memory_caveat"] = (
            "dual-number FlexAttention currently materializes a dense tangent "
            "mask/JVP path; this smoke measures that implementation"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
