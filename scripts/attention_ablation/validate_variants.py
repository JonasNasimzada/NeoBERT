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
MAX_LENGTH = 1024
EXPECTED_LAYER_PARAMETERS = 1_574_400
EXPECTED_TRAINABLE_PARAMETERS = 17_260_288
MULTISPACE_EXPECTED_LAYER_PARAMETERS = 8_504_832
MULTISPACE_EXPECTED_TRAINABLE_PARAMETERS = 99_985_152


@dataclass(frozen=True)
class Variant:
    name: str
    config_name: str
    attention_spaces: tuple[str, ...]
    attention_backends: tuple[str, ...]

    @staticmethod
    def _expand_schedule(
        schedule: tuple[str, ...],
        num_hidden_layers: int,
        schedule_name: str,
    ) -> tuple[str, ...]:
        if len(schedule) == 1:
            return schedule * num_hidden_layers
        if len(schedule) == num_hidden_layers:
            return schedule
        raise AssertionError(
            f"{schedule_name} must contain one value or one value per hidden layer; "
            f"got {len(schedule)} values for {num_hidden_layers} layers"
        )

    def space_schedule(self, num_hidden_layers: int) -> tuple[str, ...]:
        return self._expand_schedule(
            self.attention_spaces,
            num_hidden_layers,
            "attention_spaces",
        )

    def backend_schedule(self, num_hidden_layers: int) -> tuple[str, ...]:
        return self._expand_schedule(
            self.attention_backends,
            num_hidden_layers,
            "attention_backends",
        )


def _homogeneous(
    name: str,
    config_name: str,
    attention_space: str,
    attention_backend: str,
) -> Variant:
    return Variant(
        name,
        config_name,
        (attention_space,),
        (attention_backend,),
    )


VARIANTS = (
    _homogeneous(
        "complex_native", "attention-ablation-complex.yaml", "complex", "native"
    ),
    _homogeneous(
        "complex_torch", "attention-ablation-complex.yaml", "complex", "torch"
    ),
    _homogeneous(
        "complex_flash", "attention-ablation-complex.yaml", "complex", "flash"
    ),
    _homogeneous(
        "split_native", "attention-ablation-split.yaml", "split", "native"
    ),
    _homogeneous("split_torch", "attention-ablation-split.yaml", "split", "torch"),
    _homogeneous("real_torch", "attention-ablation-real.yaml", "real", "torch"),
    _homogeneous("real_flash", "attention-ablation-real.yaml", "real", "flash"),
    # Keep the original array ids stable and append the requested strict Flash
    # implementations plus the already requested dual native/Torch controls.
    _homogeneous("split_flash", "attention-ablation-split.yaml", "split", "flash"),
    _homogeneous("dual_native", "attention-ablation-dual.yaml", "dual", "native"),
    _homogeneous("dual_torch", "attention-ablation-dual.yaml", "dual", "torch"),
    _homogeneous("dual_flash", "attention-ablation-dual.yaml", "dual", "flash"),
    _homogeneous(
        "multispace_flash",
        "attention-ablation-multispace.yaml",
        "multispace",
        "flash",
    ),
)


def _model_config(variant: Variant) -> NeoBERTConfig:
    path = MODEL_CONFIG_DIR / variant.config_name
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict):
        raise TypeError(f"{path} must contain a mapping")
    num_hidden_layers = int(values.get("num_hidden_layers", 0))
    expected_spaces = variant.space_schedule(num_hidden_layers)
    expected_backends = variant.backend_schedule(num_hidden_layers)
    declared_spaces = values.get("attention_spaces")
    if declared_spaces is None:
        declared_spaces = [values.get("attention_space")] * num_hidden_layers
    if list(declared_spaces) != list(expected_spaces):
        raise AssertionError(
            f"{path} declares attention_spaces={declared_spaces!r}, "
            f"expected {list(expected_spaces)!r}"
        )
    declared_backends = values.get("attention_backends")
    if declared_backends is not None and list(declared_backends) != list(
        expected_backends
    ):
        raise AssertionError(
            f"{path} declares attention_backends={declared_backends!r}, "
            f"expected {list(expected_backends)!r}"
        )

    # Homogeneous scalar-space variants share model files and override only
    # their backend. Multispace is homogeneous at the layer-schedule level and
    # divides every layer's heads equally across the three scalar algebras.
    values["attention_space"] = expected_spaces[0]
    values["attention_backend"] = expected_backends[0]
    values["attention_spaces"] = list(expected_spaces)
    values["attention_backends"] = list(expected_backends)
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
        expected_spaces = list(variant.space_schedule(config.num_hidden_layers))
        expected_backends = list(
            variant.backend_schedule(config.num_hidden_layers)
        )
        if config.attention_spaces != expected_spaces:
            raise AssertionError(
                f"{variant.name} has attention spaces {config.attention_spaces}, "
                f"expected {expected_spaces}"
            )
        if config.attention_backends != expected_backends:
            raise AssertionError(
                f"{variant.name} has attention backends {config.attention_backends}, "
                f"expected {expected_backends}"
            )

        model = NeoBERTLMHead(config)
        if model.decoder.weight is not model.model.encoder.weight:
            raise AssertionError(
                f"{variant.name} does not tie input and output embeddings"
            )

        layer_counts = [
            sum(
                parameter.numel() * (2 if parameter.is_complex() else 1)
                for parameter in layer.parameters()
                if parameter.requires_grad
            )
            for layer in model.model.transformer_encoder
        ]
        expected_layer_parameters = (
            MULTISPACE_EXPECTED_LAYER_PARAMETERS
            if variant.name == "multispace_flash"
            else EXPECTED_LAYER_PARAMETERS
        )
        if set(layer_counts) != {expected_layer_parameters}:
            raise AssertionError(
                f"{variant.name} layer counts are {layer_counts}, "
                f"expected {expected_layer_parameters:,} each"
            )

        count = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        expected_trainable_parameters = (
            MULTISPACE_EXPECTED_TRAINABLE_PARAMETERS
            if variant.name == "multispace_flash"
            else EXPECTED_TRAINABLE_PARAMETERS
        )
        if count != expected_trainable_parameters:
            raise AssertionError(
                f"{variant.name} has {count:,} trainable parameters, "
                f"expected {expected_trainable_parameters:,}"
            )
        counts[variant.name] = count

    expected_counts = {
        variant.name: (
            MULTISPACE_EXPECTED_TRAINABLE_PARAMETERS
            if variant.name == "multispace_flash"
            else EXPECTED_TRAINABLE_PARAMETERS
        )
        for variant in VARIANTS
    }
    if counts != expected_counts:
        raise AssertionError(
            f"variant parameter counts differ from their contracts: {counts}"
        )

    if verbose:
        print(
            f"{'variant':<20} {'spaces':<37} "
            f"{'backends':<17} {'trainable':>12}"
        )
        for variant in VARIANTS:
            spaces = ">".join(variant.attention_spaces)
            backends = ">".join(variant.attention_backends)
            print(
                f"{variant.name:<20} {spaces:<37} "
                f"{backends:<17} {counts[variant.name]:>12,}"
            )

    return counts


def main() -> int:
    validate_variants()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
