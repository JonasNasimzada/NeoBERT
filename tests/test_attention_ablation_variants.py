"""Focused checks for the parameter-matched attention ablation matrix."""

from __future__ import annotations

import importlib.util
import subprocess
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
    def test_exact_seven_variant_matrix_is_parameter_matched(self):
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
                ("real", "torch"),
                ("real", "flash"),
            ],
        )

        counts = validator.validate_variants(verbose=False)

        self.assertEqual(len(counts), 7)
        self.assertEqual(
            set(counts.values()),
            {validator.EXPECTED_TRAINABLE_PARAMETERS},
        )

    def test_slurm_matrix_has_one_equal_token_step_target(self):
        common_path = NEOBERT_ROOT / "jobs" / "attention_ablation" / "common.sh"
        command = r'''
source "$1"
for task_id in {0..6}; do
    resolve_attention_variant "$task_id"
    printf '%s:%s:%s:%s\n' \
        "$ATTENTION_VARIANT" \
        "$ATTENTION_SPACE" \
        "$ATTENTION_BACKEND" \
        "$ATTENTION_TARGET_STEPS"
done
'''
        completed = subprocess.run(
            ["bash", "-c", command, "bash", str(common_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "complex-native:complex:native:84000",
                "complex-torch:complex:torch:84000",
                "complex-flash:complex:flash:84000",
                "split-native:split:native:84000",
                "split-torch:split:torch:84000",
                "real-torch:real:torch:84000",
                "real-flash:real:flash:84000",
            ],
        )

    def test_slurm_scope_validators_are_available_to_every_job_script(self):
        common_path = NEOBERT_ROOT / "jobs" / "attention_ablation" / "common.sh"
        command = r'''
source "$1"
validate_experiment_id "attention-smoke.v1"
validate_attention_seed 0
if validate_experiment_id "../escape" >/dev/null 2>&1; then exit 10; fi
if validate_attention_seed -1 >/dev/null 2>&1; then exit 11; fi
'''
        subprocess.run(
            ["bash", "-c", command, "bash", str(common_path)],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
