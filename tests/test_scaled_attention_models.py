"""Static contracts for the parameter-matched 200M and 300M MHA pairs."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    NEOBERT_ROOT
    / "scripts"
    / "attention_ablation"
    / "validate_scaled_pairs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_scaled_attention_pairs_for_tests",
    VALIDATOR_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def load_model_values(name: str) -> dict:
    path = NEOBERT_ROOT / "conf" / "model" / f"{name}.yaml"
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict):  # pragma: no cover
        raise TypeError(f"{path} must contain a mapping")
    return values


class TestScaledAttentionModelConfigs(unittest.TestCase):
    def test_four_configs_share_the_controlled_architecture(self):
        common = {
            "hidden_size": 768,
            "num_attention_heads": 12,
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
            "attention_backend": "flash",
        }
        expected_scales = {"200m": 21, "300m": 33}
        expected_variants = {
            "real": {"attention_space": "real", "intermediate_size": 4_000},
            "multispace": {
                "attention_space": "multispace",
                "intermediate_size": 2_464,
                "multispace_cuda_streams": True,
            },
        }

        for size, layers in expected_scales.items():
            for variant, variant_values in expected_variants.items():
                with self.subTest(size=size, variant=variant):
                    values = load_model_values(
                        f"attention-ablation-{variant}-{size}"
                    )
                    self.assertEqual(values["num_hidden_layers"], layers)
                    for key, expected in common.items():
                        self.assertEqual(values[key], expected)
                    for key, expected in variant_values.items():
                        self.assertEqual(values[key], expected)

                    config, path = validator._load_config(
                        validator.SCALES[size],
                        validator.VARIANTS[variant],
                    )
                    self.assertEqual(
                        path.name,
                        f"attention-ablation-{variant}-{size}.yaml",
                    )
                    self.assertEqual(config.dim_head, 64)
                    self.assertEqual(
                        config.attention_spaces,
                        [variant_values["attention_space"]] * layers,
                    )
                    self.assertEqual(
                        config.attention_backends,
                        ["flash"] * layers,
                    )

    def test_scaling_changes_only_depth_from_the_100m_pair(self):
        base_names = {
            "real": "attention-ablation-real-100m",
            "multispace": "attention-ablation-multispace",
        }
        for variant, base_name in base_names.items():
            base = load_model_values(base_name)
            self.assertEqual(base["num_hidden_layers"], 9)
            for size, expected_layers in (("200m", 21), ("300m", 33)):
                with self.subTest(variant=variant, size=size):
                    scaled = load_model_values(
                        f"attention-ablation-{variant}-{size}"
                    )
                    self.assertEqual(scaled["num_hidden_layers"], expected_layers)
                    scaled["num_hidden_layers"] = base["num_hidden_layers"]
                    self.assertEqual(scaled, base)


class TestScaledAttentionParameterFairness(unittest.TestCase):
    def test_extra_multispace_attention_is_exactly_offset_by_the_ffn(self):
        hidden_size = validator.HIDDEN_SIZE
        real_intermediate = validator.VARIANTS["real"].intermediate_size
        multispace_intermediate = validator.VARIANTS[
            "multispace"
        ].intermediate_size

        real_attention = 4 * hidden_size**2
        multispace_attention = 8 * hidden_size**2
        real_ffn = 2 * hidden_size * real_intermediate
        multispace_ffn = 2 * hidden_size * multispace_intermediate
        normalization = 2 * hidden_size

        self.assertEqual(real_attention, 2_359_296)
        self.assertEqual(multispace_attention, 4_718_592)
        self.assertEqual(real_ffn, 6_144_000)
        self.assertEqual(multispace_ffn, 3_784_704)
        self.assertEqual(
            multispace_attention - real_attention,
            real_ffn - multispace_ffn,
        )
        self.assertEqual(
            real_attention + real_ffn + normalization,
            validator.EXPECTED_BLOCK_PARAMETERS,
        )
        self.assertEqual(
            multispace_attention + multispace_ffn + normalization,
            validator.EXPECTED_BLOCK_PARAMETERS,
        )

    def test_depth_formula_gives_exact_matched_totals(self):
        shared_parameters = (
            validator.VOCAB_SIZE * validator.HIDDEN_SIZE
            + validator.EXPECTED_FINAL_NORM_PARAMETERS
        )
        self.assertEqual(shared_parameters, 23_441_664)

        for size, contract in validator.SCALES.items():
            with self.subTest(size=size):
                expected = (
                    shared_parameters
                    + contract.layers * validator.EXPECTED_BLOCK_PARAMETERS
                )
                self.assertEqual(expected, contract.expected_parameters)
                self.assertLess(
                    abs(expected - int(size.removesuffix("m")) * 1_000_000),
                    5_000_000,
                )

    def test_validator_builds_all_production_graphs_on_meta(self):
        model_class = validator.NeoBERTLMHead
        allocation_devices = []

        def record_allocation(config):
            model = model_class(config)
            allocation_devices.append(next(model.parameters()).device.type)
            return model

        with mock.patch.object(
            validator,
            "NeoBERTLMHead",
            side_effect=record_allocation,
        ):
            report = validator.validate_scaled_pairs(verbose=False)

        self.assertEqual(allocation_devices, ["meta"] * 4)
        pairs = report["pairs"]
        self.assertEqual(set(pairs), {"200m", "300m"})
        for size, contract in validator.SCALES.items():
            with self.subTest(size=size):
                pair = pairs[size]
                self.assertTrue(pair["pair_exactly_matched"])
                self.assertEqual(
                    pair["exact_parameters_each"],
                    contract.expected_parameters,
                )
                real = pair["variants"]["real"]
                multispace = pair["variants"]["multispace"]
                self.assertEqual(
                    real["trainable_real_scalar_parameters"],
                    multispace["trainable_real_scalar_parameters"],
                )
                self.assertEqual(real["heads_by_space"], {"real": 12})
                self.assertEqual(
                    multispace["heads_by_space"],
                    {"complex": 4, "split": 4, "dual": 4},
                )
                self.assertEqual(
                    real["parameters_per_block"],
                    multispace["parameters_per_block"],
                )


if __name__ == "__main__":
    unittest.main()
