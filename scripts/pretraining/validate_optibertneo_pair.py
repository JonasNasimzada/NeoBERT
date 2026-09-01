#!/usr/bin/env python3
"""Validate the parameter-matched real and multispace OptiBERTneo pair."""

from __future__ import annotations

import sys
from dataclasses import dataclass
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


MODEL_CONFIG_DIR = PROJECT_ROOT / "conf" / "model"
VOCAB_SIZE = 50_265
PAD_TOKEN_ID = 1
MAX_LENGTH = 1_024
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 28
NUM_ATTENTION_HEADS = 12
HEADS_PER_SPACE = 4
EXPECTED_NORM_PARAMETERS = 1_536
EXPECTED_BLOCK_PARAMETERS = 7_079_424
EXPECTED_EMBEDDING_PARAMETERS = 38_603_520
EXPECTED_NON_EMBEDDING_PARAMETERS = 198_225_408
EXPECTED_TOTAL_PARAMETERS = 236_828_928


@dataclass(frozen=True)
class ModelContract:
    name: str
    config_filename: str
    attention_space: str
    nominal_intermediate_size: int
    effective_intermediate_size: int
    attention_parameters: int
    ffn_parameters: int


CONTRACTS = (
    ModelContract(
        name="real",
        config_filename="optibertneo-198m.yaml",
        attention_space="real",
        nominal_intermediate_size=3_072,
        effective_intermediate_size=2_048,
        attention_parameters=2_359_296,
        ffn_parameters=4_718_592,
    ),
    ModelContract(
        name="multispace",
        config_filename="optibertneo-198m-multispace.yaml",
        attention_space="multispace",
        nominal_intermediate_size=1_536,
        effective_intermediate_size=1_024,
        attention_parameters=4_718_592,
        ffn_parameters=2_359_296,
    ),
)


def _scalar_parameter_count(module) -> int:
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


def _load_config(contract: ModelContract) -> NeoBERTConfig:
    path = MODEL_CONFIG_DIR / contract.config_filename
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict):
        raise TypeError(f"{path} must contain a mapping")

    expected_values = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "intermediate_size": contract.nominal_intermediate_size,
        "rope": True,
        "rms_norm": True,
        "embedding_rms_norm": True,
        "hidden_act": "swiglu",
        "fused_swiglu": False,
        "dropout": 0,
        "attention_dropout": 0,
        "tie_word_embeddings": True,
        "lm_head_bias": False,
        "attention_space": contract.attention_space,
        "attention_backend": "flex",
    }
    for key, expected in expected_values.items():
        actual = values.get(key)
        if actual != expected:
            raise AssertionError(
                f"{path} declares {key}={actual!r}, expected {expected!r}"
            )
    if bool(values.get("ngpt", False)):
        raise AssertionError(f"{path} must select standard pre-RMSNorm blocks")

    values.update(
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_TOKEN_ID,
        max_length=MAX_LENGTH,
    )
    return NeoBERTConfig(**values)


def _attention_parameter_count(layer, attention_space: str) -> int:
    if attention_space == "real":
        if (
            layer.qkv is None
            or layer.wo is None
            or layer.complex_attention is not None
        ):
            raise AssertionError("real layer has an unexpected attention layout")
        return _scalar_parameter_count(layer.qkv) + _scalar_parameter_count(layer.wo)

    if attention_space == "multispace":
        if (
            layer.qkv is not None
            or layer.wo is not None
            or layer.complex_attention is None
        ):
            raise AssertionError("multispace layer has an unexpected attention layout")
        attention = layer.complex_attention
        if tuple(attention.space_names) != ("complex", "split", "dual"):
            raise AssertionError(
                f"unexpected multispace groups: {attention.space_names!r}"
            )
        heads_by_space = {
            space: attention.heads_per_space for space in attention.space_names
        }
        expected_heads = {
            "complex": HEADS_PER_SPACE,
            "split": HEADS_PER_SPACE,
            "dual": HEADS_PER_SPACE,
        }
        if heads_by_space != expected_heads:
            raise AssertionError(
                f"multispace heads are {heads_by_space!r}, expected {expected_heads!r}"
            )
        return _scalar_parameter_count(attention)

    raise AssertionError(f"unsupported OptiBERTneo attention space: {attention_space}")


def _assert_bias_free(model, name: str) -> None:
    biased_modules = [
        module_name
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module.bias is not None
    ]
    if biased_modules:
        raise AssertionError(
            f"{name} has biased linear modules: {biased_modules!r}"
        )


def validate_optibertneo_pair(*, verbose: bool = True) -> dict[str, int]:
    totals: dict[str, int] = {}

    for contract in CONTRACTS:
        config = _load_config(contract)
        expected_spaces = [contract.attention_space] * NUM_HIDDEN_LAYERS
        expected_backends = ["flex"] * NUM_HIDDEN_LAYERS
        if config.attention_spaces != expected_spaces:
            raise AssertionError(
                f"{contract.name} schedule is {config.attention_spaces!r}, "
                f"expected {expected_spaces!r}"
            )
        if config.attention_backends != expected_backends:
            raise AssertionError(
                f"{contract.name} backends are {config.attention_backends!r}, "
                f"expected {expected_backends!r}"
            )

        # Meta construction validates the actual module graph without allocating
        # approximately one gigabyte of fp32 storage for each model.
        with torch.device("meta"):
            model = NeoBERTLMHead(config)

        if model.decoder.weight is not model.model.encoder.weight:
            raise AssertionError(
                f"{contract.name} does not tie input/output embeddings"
            )
        _assert_bias_free(model, contract.name)

        layers = model.model.transformer_encoder
        if len(layers) != NUM_HIDDEN_LAYERS:
            raise AssertionError(
                f"{contract.name} has {len(layers)} blocks, "
                f"expected {NUM_HIDDEN_LAYERS}"
            )

        for layer_index, layer in enumerate(layers):
            attention_count = _attention_parameter_count(
                layer, contract.attention_space
            )
            ffn_count = _scalar_parameter_count(layer.ffn)
            norm_count = (
                _scalar_parameter_count(layer.attention_norm)
                + _scalar_parameter_count(layer.ffn_norm)
            )
            block_count = _scalar_parameter_count(layer)
            effective_width = layer.ffn.w3.in_features

            expected_counts = {
                "attention": (attention_count, contract.attention_parameters),
                "ffn": (ffn_count, contract.ffn_parameters),
                "norm": (norm_count, EXPECTED_NORM_PARAMETERS),
                "block": (block_count, EXPECTED_BLOCK_PARAMETERS),
                "effective SwiGLU width": (
                    effective_width,
                    contract.effective_intermediate_size,
                ),
            }
            for label, (actual, expected) in expected_counts.items():
                if actual != expected:
                    raise AssertionError(
                        f"{contract.name} layer {layer_index} {label} is "
                        f"{actual:,}, expected {expected:,}"
                    )

        embedding_count = _scalar_parameter_count(model.model.encoder)
        total = _scalar_parameter_count(model)
        non_embedding = total - embedding_count
        if embedding_count != EXPECTED_EMBEDDING_PARAMETERS:
            raise AssertionError(
                f"{contract.name} has {embedding_count:,} embedding parameters, "
                f"expected {EXPECTED_EMBEDDING_PARAMETERS:,}"
            )
        if non_embedding != EXPECTED_NON_EMBEDDING_PARAMETERS:
            raise AssertionError(
                f"{contract.name} has {non_embedding:,} non-embedding parameters, "
                f"expected {EXPECTED_NON_EMBEDDING_PARAMETERS:,}"
            )
        if total != EXPECTED_TOTAL_PARAMETERS:
            raise AssertionError(
                f"{contract.name} has {total:,} total parameters, "
                f"expected {EXPECTED_TOTAL_PARAMETERS:,}"
            )
        totals[contract.name] = total

    if len(set(totals.values())) != 1:
        raise AssertionError(f"paired model totals do not match: {totals!r}")

    if verbose:
        print(
            f"{'model':<12} {'heads':>9} {'attention':>14} {'ffn':>14} "
            f"{'block':>14} {'non-embedding':>16} {'total':>14}"
        )
        for contract in CONTRACTS:
            heads = "12 real" if contract.attention_space == "real" else "4/4/4"
            print(
                f"{contract.name:<12} {heads:>9} "
                f"{contract.attention_parameters:>14,} "
                f"{contract.ffn_parameters:>14,} "
                f"{EXPECTED_BLOCK_PARAMETERS:>14,} "
                f"{EXPECTED_NON_EMBEDDING_PARAMETERS:>16,} "
                f"{totals[contract.name]:>14,}"
            )

    return totals


def main() -> int:
    validate_optibertneo_pair()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
