"""Focused checks for the controlled attention experiment matrix."""

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
    def test_exact_twelve_variant_matrix_has_declared_parameter_budgets(self):
        self.assertEqual(
            [
                (variant.attention_spaces, variant.attention_backends)
                for variant in validator.VARIANTS
            ],
            [
                (("complex",), ("native",)),
                (("complex",), ("torch",)),
                (("complex",), ("flash",)),
                (("split",), ("native",)),
                (("split",), ("torch",)),
                (("real",), ("torch",)),
                (("real",), ("flash",)),
                (("split",), ("flash",)),
                (("dual",), ("native",)),
                (("dual",), ("torch",)),
                (("dual",), ("flash",)),
                (("multispace",), ("flash",)),
            ],
        )

        counts = validator.validate_variants(verbose=False)

        self.assertEqual(len(counts), 12)
        self.assertEqual(
            {
                count
                for name, count in counts.items()
                if name != "multispace_flash"
            },
            {validator.EXPECTED_TRAINABLE_PARAMETERS},
        )
        self.assertEqual(
            counts["multispace_flash"],
            validator.MULTISPACE_EXPECTED_TRAINABLE_PARAMETERS,
        )
        self.assertEqual(
            validator.MULTISPACE_EXPECTED_LAYER_PARAMETERS,
            8_504_832,
        )
        self.assertEqual(
            validator.MULTISPACE_EXPECTED_TRAINABLE_PARAMETERS,
            99_985_152,
        )

    def test_slurm_matrix_has_one_equal_token_step_target(self):
        common_path = NEOBERT_ROOT / "jobs" / "attention_ablation" / "common.sh"
        command = r'''
source "$1"
for task_id in {0..11}; do
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
                "split-flash:split:flash:84000",
                "dual-native:dual:native:84000",
                "dual-torch:dual:torch:84000",
                "dual-flash:dual:flash:84000",
                "multispace-flash:multispace:flash:84000",
            ],
        )

    def test_multispace_task_uses_the_explicit_model_config(self):
        common_path = NEOBERT_ROOT / "jobs" / "attention_ablation" / "common.sh"
        command = r'''
source "$1"
resolve_attention_variant 11
printf '%s:%s:%s:%s\n' \
    "$ATTENTION_VARIANT" \
    "$ATTENTION_SPACE" \
    "$ATTENTION_BACKEND" \
    "$MODEL_CONFIG"
'''
        completed = subprocess.run(
            ["bash", "-c", command, "bash", str(common_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.stdout.strip(),
            "multispace-flash:multispace:flash:attention-ablation-multispace",
        )
        config = validator._model_config(validator.VARIANTS[-1])
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_hidden_layers, 9)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.dim_head, 64)
        self.assertEqual(config.intermediate_size, 2464)
        self.assertEqual(config.attention_spaces, ["multispace"] * 9)
        self.assertEqual(config.attention_backends, ["flash"] * 9)

    def test_slurm_arrays_include_multispace_only_for_model_level_jobs(self):
        jobs = NEOBERT_ROOT / "jobs" / "attention_ablation"
        train = (jobs / "train.sbatch").read_text(encoding="utf-8")
        benchmark = (jobs / "benchmark.sbatch").read_text(encoding="utf-8")
        paper = (jobs / "benchmark_attention_papers.sbatch").read_text(
            encoding="utf-8"
        )
        submit = (jobs / "submit.sh").read_text(encoding="utf-8")

        self.assertIn("#SBATCH --array=0-11", train)
        self.assertIn("#SBATCH --array=0-11", benchmark)
        self.assertIn("#SBATCH --array=0-10", paper)
        self.assertEqual(submit.count("--array=0-11"), 2)
        self.assertEqual(submit.count("--array=0-10"), 1)

    def test_1024_geometry_preserves_the_controlled_token_budget(self):
        common_path = NEOBERT_ROOT / "jobs" / "attention_ablation" / "common.sh"
        command = r'''
source "$1"
printf '%s:%s:%s:%s:%s\n' \
    "$ATTENTION_SEQUENCE_LENGTH" \
    "$ATTENTION_EFFECTIVE_SEQUENCE_BATCH" \
    "$ATTENTION_TOKEN_POSITIONS_PER_STEP" \
    "$ATTENTION_EQUAL_TOKEN_STEPS" \
    "$ATTENTION_TOTAL_TOKEN_POSITIONS"
'''
        completed = subprocess.run(
            ["bash", "-c", command, "bash", str(common_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.stdout.strip(),
            "1024:16:16384:84000:1376256000",
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
