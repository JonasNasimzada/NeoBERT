"""Focused checks for the parameter-matched attention ablation matrix."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    NEOBERT_ROOT
    / "scripts"
    / "attention_ablation"
    / "validate_variants.py"
)
SPEC = importlib.util.spec_from_file_location(
    "attention_ablation_validator_for_tests",
    VALIDATOR_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


class TestAttentionAblationVariants(unittest.TestCase):
    def test_exact_nine_variant_matrix_is_parameter_matched(self):
        self.assertEqual(
            [
                (variant.attention_space, variant.attention_backend)
                for variant in validator.VARIANTS
            ],
            [
                ("complex", "native"),
                ("complex", "torch"),
                ("complex", "flash"),
                ("split", "native"),
                ("split", "torch"),
                ("dual", "native"),
                ("dual", "torch"),
                ("real", "torch"),
                ("real", "flash"),
            ],
        )

        counts = validator.validate_variants(verbose=False)

        self.assertEqual(len(counts), 9)
        self.assertEqual(
            set(counts.values()),
            {validator.EXPECTED_TRAINABLE_PARAMETERS},
        )


if __name__ == "__main__":
    unittest.main()
