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

    def test_slurm_matrix_has_backend_calibrated_step_targets(self):
        common_path = NEOBERT_ROOT / "jobs" / "attention_ablation" / "common.sh"
        command = r'''
source "$1"
for task_id in {0..8}; do
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
                "complex-native:complex:native:95000",
                "complex-torch:complex:torch:113500",
                "complex-flash:complex:flash:96000",
                "split-native:split:native:66500",
                "split-torch:split:torch:89000",
                "dual-native:dual:native:52000",
                "dual-torch:dual:torch:42000",
                "real-torch:real:torch:156000",
                "real-flash:real:flash:153000",
            ],
        )

    def test_dual_flex_is_available_only_as_an_uncalibrated_variant(self):
        common_path = NEOBERT_ROOT / "jobs" / "attention_ablation" / "common.sh"
        command = r'''
source "$1"
resolve_attention_variant 9
printf '%s:%s:%s:%s:%s\n' \
    "$ATTENTION_VARIANT" \
    "$ATTENTION_SPACE" \
    "$ATTENTION_BACKEND" \
    "$MODEL_CONFIG" \
    "$ATTENTION_TARGET_STEPS"
'''
        completed = subprocess.run(
            ["bash", "-c", command, "bash", str(common_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.stdout.strip(),
            "dual-flex:dual:flex:attention-ablation-dual:",
        )

    def test_slurm_scope_validators_are_available_to_every_job_script(self):
        common_path = NEOBERT_ROOT / "jobs" / "attention_ablation" / "common.sh"
        command = r'''
source "$1"
validate_experiment_id "dual-flex-smoke.v1"
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
