#!/usr/bin/env python3
"""Validate the parameter-matched 200M and 300M real/multispace pairs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_IMPORT_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT.parent)
for import_root in reversed(LOCAL_IMPORT_ROOTS):
    import_root_string = str(import_root)
    if import_root_string not in sys.path:
        sys.path.insert(0, import_root_string)

from neobert.model import NeoBERTConfig, NeoBERTLMHead


MODEL_CONFIG_DIR = PROJECT_ROOT / "conf" / "model"
VOCAB_SIZE = 30_522
PAD_TOKEN_ID = 0
MAX_LENGTH = 1_024
HIDDEN_SIZE = 768
NUM_ATTENTION_HEADS = 12
HEAD_DIMENSION = 64
HEADS_PER_SPACE = 4
EXPECTED_EMBEDDING_PARAMETERS = 23_440_896
EXPECTED_FINAL_NORM_PARAMETERS = 768
EXPECTED_NORM_PARAMETERS_PER_BLOCK = 1_536
EXPECTED_BLOCK_PARAMETERS = 8_504_832


@dataclass(frozen=True)
class ScaleContract:
    name: str
    layers: int
    expected_parameters: int


@dataclass(frozen=True)
class VariantContract:
    name: str
    attention_space: str
    intermediate_size: int
    attention_parameters: int
    ffn_parameters: int


SCALES = {
    "200m": ScaleContract("200m", layers=21, expected_parameters=202_043_136),
    "300m": ScaleContract("300m", layers=33, expected_parameters=304_101_120),
}

VARIANTS = {
    "real": VariantContract(
        name="real",
        attention_space="real",
        intermediate_size=4_000,
        attention_parameters=2_359_296,
        ffn_parameters=6_144_000,
    ),
    "multispace": VariantContract(
        name="multispace",
        attention_space="multispace",
        intermediate_size=2_464,
        attention_parameters=4_718_592,
        ffn_parameters=3_784_704,
    ),
}


def _scalar_parameter_count(module: torch.nn.Module) -> int:
    """Count unique trainable real scalars, including complex components."""
    unique_parameters = {
        id(parameter): parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in unique_parameters.values()
    )


def _config_path(scale: str, variant: str) -> Path:
    return MODEL_CONFIG_DIR / f"attention-ablation-{variant}-{scale}.yaml"


def _load_config(
    scale: ScaleContract,
    variant: VariantContract,
) -> tuple[NeoBERTConfig, Path]:
    path = _config_path(scale.name, variant.name)
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict):
        raise TypeError(f"{path} must contain a mapping")

    expected_values = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": scale.layers,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "intermediate_size": variant.intermediate_size,
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
        "attention_space": variant.attention_space,
        "attention_backend": "flash",
    }
    for key, expected in expected_values.items():
        actual = values.get(key)
        if actual != expected:
            raise AssertionError(
                f"{path} declares {key}={actual!r}, expected {expected!r}"
            )
    if variant.name == "multispace" and values.get("multispace_cuda_streams") is not True:
        raise AssertionError(f"{path} must enable multispace_cuda_streams")

    values.update(
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_TOKEN_ID,
        max_length=MAX_LENGTH,
    )
    return NeoBERTConfig(**values), path


def _attention_parameter_count(
    layer: torch.nn.Module,
    variant: VariantContract,
) -> tuple[int, dict[str, int]]:
    if variant.name == "real":
        if layer.qkv is None or layer.wo is None or layer.complex_attention is not None:
            raise AssertionError("real-MHA layer has an unexpected attention layout")
        count = _scalar_parameter_count(layer.qkv) + _scalar_parameter_count(layer.wo)
        return count, {"real": NUM_ATTENTION_HEADS}

    if layer.qkv is not None or layer.wo is not None or layer.complex_attention is None:
        raise AssertionError("multispace layer has an unexpected attention layout")
    attention = layer.complex_attention
    if tuple(attention.space_names) != ("complex", "split", "dual"):
        raise AssertionError(f"unexpected multispace groups: {attention.space_names!r}")
    heads = {
        space: int(attention.heads_per_space) for space in attention.space_names
    }
    expected_heads = {
        "complex": HEADS_PER_SPACE,
        "split": HEADS_PER_SPACE,
        "dual": HEADS_PER_SPACE,
    }
    if heads != expected_heads:
        raise AssertionError(f"multispace heads are {heads!r}, expected {expected_heads!r}")
    if attention.out_proj.in_features != 2 * HIDDEN_SIZE:
        raise AssertionError("multispace output projection must retain both components")
    return _scalar_parameter_count(attention), heads


def _assert_bias_free(model: torch.nn.Module, name: str) -> None:
    biased_modules = [
        module_name
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module.bias is not None
    ]
    if biased_modules:
        raise AssertionError(f"{name} has biased linear modules: {biased_modules!r}")


def _require_a100() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; an A100 allocation is required")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "A100" not in name.upper() or capability != (8, 0):
        raise RuntimeError(f"expected an NVIDIA A100 (SM80), found {name} {capability}")
    return {
        "name": name,
        "capability": list(capability),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def validate_scaled_pairs(
    selected_sizes: Iterable[str] = ("200m", "300m"),
    *,
    require_a100: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    selected = tuple(selected_sizes)
    if not selected:
        raise ValueError("at least one model size must be selected")
    unknown = sorted(set(selected) - set(SCALES))
    if unknown:
        raise ValueError(f"unknown model sizes: {unknown!r}")

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "vocab_size": VOCAB_SIZE,
        "max_length": MAX_LENGTH,
        "gpu": _require_a100() if require_a100 else None,
        "pairs": {},
    }
    pairs = report["pairs"]
    assert isinstance(pairs, dict)

    for size_name in selected:
        scale = SCALES[size_name]
        variant_reports: dict[str, object] = {}
        totals: dict[str, int] = {}

        for variant in VARIANTS.values():
            config, path = _load_config(scale, variant)
            expected_spaces = [variant.attention_space] * scale.layers
            expected_backends = ["flash"] * scale.layers
            if config.attention_spaces != expected_spaces:
                raise AssertionError(
                    f"{variant.name}/{size_name} space schedule is invalid"
                )
            if config.attention_backends != expected_backends:
                raise AssertionError(
                    f"{variant.name}/{size_name} backend schedule is invalid"
                )
            if config.dim_head != HEAD_DIMENSION:
                raise AssertionError(
                    f"{variant.name}/{size_name} head dimension is {config.dim_head}"
                )

            # Validate the complete production graph without allocating more than
            # a gigabyte of FP32 parameter storage per model during this pass.
            with torch.device("meta"):
                model = NeoBERTLMHead(config)

            if model.decoder.weight is not model.model.encoder.weight:
                raise AssertionError(
                    f"{variant.name}/{size_name} does not tie input/output embeddings"
                )
            _assert_bias_free(model, f"{variant.name}/{size_name}")

            layers = model.model.transformer_encoder
            if len(layers) != scale.layers:
                raise AssertionError(
                    f"{variant.name}/{size_name} has {len(layers)} blocks, "
                    f"expected {scale.layers}"
                )

            layer_reports = []
            for layer_index, layer in enumerate(layers):
                attention_count, heads = _attention_parameter_count(layer, variant)
                ffn_count = _scalar_parameter_count(layer.ffn)
                norm_count = (
                    _scalar_parameter_count(layer.attention_norm)
                    + _scalar_parameter_count(layer.ffn_norm)
                )
                block_count = _scalar_parameter_count(layer)
                actual_expected = {
                    "attention": (attention_count, variant.attention_parameters),
                    "ffn": (ffn_count, variant.ffn_parameters),
                    "norm": (norm_count, EXPECTED_NORM_PARAMETERS_PER_BLOCK),
                    "block": (block_count, EXPECTED_BLOCK_PARAMETERS),
                }
                for label, (actual, expected) in actual_expected.items():
                    if actual != expected:
                        raise AssertionError(
                            f"{variant.name}/{size_name} layer {layer_index} "
                            f"{label} is {actual:,}, expected {expected:,}"
                        )
                layer_reports.append(block_count)

            embedding_count = _scalar_parameter_count(model.model.encoder)
            final_norm_count = _scalar_parameter_count(model.model.layer_norm)
            total = _scalar_parameter_count(model)
            if embedding_count != EXPECTED_EMBEDDING_PARAMETERS:
                raise AssertionError(
                    f"{variant.name}/{size_name} embedding count is "
                    f"{embedding_count:,}, expected {EXPECTED_EMBEDDING_PARAMETERS:,}"
                )
            if final_norm_count != EXPECTED_FINAL_NORM_PARAMETERS:
                raise AssertionError(
                    f"{variant.name}/{size_name} final norm count is "
                    f"{final_norm_count:,}, expected {EXPECTED_FINAL_NORM_PARAMETERS:,}"
                )
            if total != scale.expected_parameters:
                raise AssertionError(
                    f"{variant.name}/{size_name} has {total:,} parameters, "
                    f"expected {scale.expected_parameters:,}"
                )
            totals[variant.name] = total
            variant_reports[variant.name] = {
                "config": str(path),
                "attention_space": variant.attention_space,
                "attention_backend": "flash",
                "hidden_size": HIDDEN_SIZE,
                "layers": scale.layers,
                "heads": NUM_ATTENTION_HEADS,
                "head_dimension": HEAD_DIMENSION,
                "heads_by_space": heads,
                "intermediate_size": variant.intermediate_size,
                "attention_parameters_per_block": variant.attention_parameters,
                "ffn_parameters_per_block": variant.ffn_parameters,
                "parameters_per_block": layer_reports[0],
                "embedding_parameters": embedding_count,
                "final_norm_parameters": final_norm_count,
                "trainable_real_scalar_parameters": total,
            }

        if len(set(totals.values())) != 1:
            raise AssertionError(f"{size_name} pair is not matched: {totals!r}")
        pairs[size_name] = {
            "target_parameters": int(size_name.removesuffix("m")) * 1_000_000,
            "exact_parameters_each": scale.expected_parameters,
            "pair_exactly_matched": True,
            "variants": variant_reports,
        }

    if verbose:
        print(
            f"{'size':<7} {'variant':<12} {'layers':>7} {'heads':>9} "
            f"{'ffn':>8} {'per block':>14} {'total':>14}"
        )
        for size_name in selected:
            pair = pairs[size_name]
            assert isinstance(pair, dict)
            variants = pair["variants"]
            assert isinstance(variants, dict)
            for variant_name in ("real", "multispace"):
                item = variants[variant_name]
                assert isinstance(item, dict)
                heads = "12 real" if variant_name == "real" else "4/4/4"
                print(
                    f"{size_name:<7} {variant_name:<12} {item['layers']:>7} "
                    f"{heads:>9} {item['intermediate_size']:>8,} "
                    f"{item['parameters_per_block']:>14,} "
                    f"{item['trainable_real_scalar_parameters']:>14,}"
                )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        dest="sizes",
        action="append",
        choices=tuple(SCALES),
        help="validate one size (repeatable; default: both)",
    )
    parser.add_argument(
        "--require-a100",
        action="store_true",
        help="fail unless running inside an NVIDIA A100 SM80 allocation",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_scaled_pairs(
        args.sizes or tuple(SCALES),
        require_a100=args.require_a100,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
