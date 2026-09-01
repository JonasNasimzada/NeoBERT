#!/usr/bin/env python3
"""Validate the exactly parameter-matched multispace and real-MHA pair."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_IMPORT_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT.parent)
for import_root in reversed(LOCAL_IMPORT_ROOTS):
    import_root_string = str(import_root)
    if import_root_string not in sys.path:
        sys.path.insert(0, import_root_string)

from omegaconf import OmegaConf

from neobert.model import NeoBERTConfig, NeoBERTLMHead


MODEL_CONFIG_DIR = PROJECT_ROOT / "conf" / "model"
VOCAB_SIZE = 30_522
MAX_LENGTH = 1_024
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 9
NUM_ATTENTION_HEADS = 12
EXPECTED_BLOCK_PARAMETERS = 8_504_832
EXPECTED_TRAINABLE_PARAMETERS = 99_985_152


@dataclass(frozen=True)
class ModelContract:
    name: str
    config_filename: str
    attention_space: str
    intermediate_size: int
    attention_parameters: int


CONTRACTS = (
    ModelContract(
        name="multispace",
        config_filename="attention-ablation-multispace.yaml",
        attention_space="multispace",
        intermediate_size=2_464,
        attention_parameters=4_718_592,
    ),
    ModelContract(
        name="real",
        config_filename="attention-ablation-real-100m.yaml",
        attention_space="real",
        intermediate_size=4_000,
        attention_parameters=2_359_296,
    ),
)


def _scalar_parameter_count(module) -> int:
    """Count trainable real scalars, including both parts of complex tensors."""
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in module.parameters()
        if parameter.requires_grad
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
        "intermediate_size": contract.intermediate_size,
        "hidden_act": "gelu",
        "attention_space": contract.attention_space,
        "attention_backend": "flash",
        "tie_word_embeddings": True,
        "lm_head_bias": False,
        "ngpt": False,
    }
    for key, expected in expected_values.items():
        actual = values.get(key)
        if actual != expected:
            raise AssertionError(
                f"{path} declares {key}={actual!r}, expected {expected!r}"
            )

    values.update(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        pad_token_id=0,
    )
    return NeoBERTConfig(**values)


def _attention_parameter_count(layer, attention_space: str) -> int:
    if attention_space == "real":
        if (
            layer.qkv is None
            or layer.wo is None
            or layer.complex_attention is not None
        ):
            raise AssertionError("real-MHA layer has an unexpected attention layout")
        return _scalar_parameter_count(layer.qkv) + _scalar_parameter_count(layer.wo)

    if attention_space == "multispace":
        if (
            layer.qkv is not None
            or layer.wo is not None
            or layer.complex_attention is None
        ):
            raise AssertionError("multispace layer has an unexpected attention layout")
        return _scalar_parameter_count(layer.complex_attention)

    raise AssertionError(f"unsupported paired attention space: {attention_space}")


def validate_100m_pair(*, verbose: bool = True) -> dict[str, int]:
    counts: dict[str, int] = {}

    for contract in CONTRACTS:
        config = _load_config(contract)
        expected_spaces = [contract.attention_space] * NUM_HIDDEN_LAYERS
        expected_backends = ["flash"] * NUM_HIDDEN_LAYERS
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

        model = NeoBERTLMHead(config)
        if model.decoder.weight is not model.model.encoder.weight:
            raise AssertionError(
                f"{contract.name} does not tie input/output embeddings"
            )

        layers = model.model.transformer_encoder
        if len(layers) != NUM_HIDDEN_LAYERS:
            raise AssertionError(
                f"{contract.name} has {len(layers)} blocks, expected {NUM_HIDDEN_LAYERS}"
            )

        block_counts = [_scalar_parameter_count(layer) for layer in layers]
        if set(block_counts) != {EXPECTED_BLOCK_PARAMETERS}:
            raise AssertionError(
                f"{contract.name} block counts are {block_counts!r}, expected "
                f"{EXPECTED_BLOCK_PARAMETERS:,} each"
            )

        attention_counts = [
            _attention_parameter_count(layer, contract.attention_space)
            for layer in layers
        ]
        if set(attention_counts) != {contract.attention_parameters}:
            raise AssertionError(
                f"{contract.name} attention counts are {attention_counts!r}, "
                f"expected {contract.attention_parameters:,} each"
            )

        total = _scalar_parameter_count(model)
        if total != EXPECTED_TRAINABLE_PARAMETERS:
            raise AssertionError(
                f"{contract.name} has {total:,} trainable parameters, "
                f"expected {EXPECTED_TRAINABLE_PARAMETERS:,}"
            )
        counts[contract.name] = total

    if len(set(counts.values())) != 1:
        raise AssertionError(f"paired model totals do not match: {counts!r}")

    if verbose:
        print(f"{'model':<12} {'attention/block':>18} {'block':>14} {'total':>14}")
        for contract in CONTRACTS:
            print(
                f"{contract.name:<12} {contract.attention_parameters:>18,} "
                f"{EXPECTED_BLOCK_PARAMETERS:>14,} "
                f"{counts[contract.name]:>14,}"
            )

    return counts


def main() -> int:
    validate_100m_pair()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
