#!/usr/bin/env python3
"""Validate the exact parameter budget for the attention ablation matrix."""

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
MAX_LENGTH = 512
EXPECTED_LAYER_PARAMETERS = 1_574_400
EXPECTED_TRAINABLE_PARAMETERS = 17_260_288


@dataclass(frozen=True)
class Variant:
    name: str
    config_name: str
    attention_space: str
    attention_backend: str


VARIANTS = (
    Variant("complex_native", "attention-ablation-complex.yaml", "complex", "native"),
    Variant("complex_torch", "attention-ablation-complex.yaml", "complex", "torch"),
    Variant("complex_flash", "attention-ablation-complex.yaml", "complex", "flash"),
    Variant("split_native", "attention-ablation-split.yaml", "split", "native"),
    Variant("split_torch", "attention-ablation-split.yaml", "split", "torch"),
    Variant("real_torch", "attention-ablation-real.yaml", "real", "torch"),
    Variant("real_flash", "attention-ablation-real.yaml", "real", "flash"),
)


def _model_config(variant: Variant) -> NeoBERTConfig:
    path = MODEL_CONFIG_DIR / variant.config_name
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict):
        raise TypeError(f"{path} must contain a mapping")
    if values.get("attention_space") != variant.attention_space:
        raise AssertionError(
            f"{path} declares attention_space={values.get('attention_space')!r}, "
            f"expected {variant.attention_space!r}"
        )

    # The backend is the sole per-run override. Vocabulary and sequence length
    # normally arrive through Hydra's tokenizer group during training.
    values["attention_backend"] = variant.attention_backend
    values.update(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        pad_token_id=0,
    )
    return NeoBERTConfig(**values)


def validate_variants(*, verbose: bool = True) -> dict[str, int]:
    counts: dict[str, int] = {}

    for variant in VARIANTS:
        config = _model_config(variant)
        if config.attention_spaces != [variant.attention_space] * config.num_hidden_layers:
            raise AssertionError(f"{variant.name} is not homogeneous in attention space")
        if config.attention_backends != [variant.attention_backend] * config.num_hidden_layers:
            raise AssertionError(f"{variant.name} is not homogeneous in attention backend")

        model = NeoBERTLMHead(config)
        if model.decoder.weight is not model.model.encoder.weight:
            raise AssertionError(f"{variant.name} does not tie input and output embeddings")

        layer_counts = [
            sum(parameter.numel() for parameter in layer.parameters() if parameter.requires_grad)
            for layer in model.model.transformer_encoder
        ]
        if set(layer_counts) != {EXPECTED_LAYER_PARAMETERS}:
            raise AssertionError(
                f"{variant.name} layer counts are {layer_counts}, "
                f"expected {EXPECTED_LAYER_PARAMETERS:,} each"
            )

        count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if count != EXPECTED_TRAINABLE_PARAMETERS:
            raise AssertionError(
                f"{variant.name} has {count:,} trainable parameters, "
                f"expected {EXPECTED_TRAINABLE_PARAMETERS:,}"
            )
        counts[variant.name] = count

    if set(counts.values()) != {EXPECTED_TRAINABLE_PARAMETERS}:
        raise AssertionError(f"variant parameter counts differ: {counts}")

    if verbose:
        print(f"{'variant':<20} {'space':<8} {'backend':<7} {'trainable':>12}")
        for variant in VARIANTS:
            print(
                f"{variant.name:<20} {variant.attention_space:<8} "
                f"{variant.attention_backend:<7} {counts[variant.name]:>12,}"
            )

    return counts


def main() -> int:
    validate_variants()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
