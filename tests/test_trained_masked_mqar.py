"""Protocol and launcher tests for the trained masked-MQAR transfer probe."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = NEOBERT_ROOT / "scripts" / "attention_ablation"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


def load_script(name: str):
    path = SCRIPT_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_for_tests", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trained = load_script("trained_masked_mqar")


class TestSplitIsolation(unittest.TestCase):
    def test_split_seeds_are_deterministic_and_disjoint(self):
        first = {
            split: trained.split_seed(42, split)
            for split in ("train", "validation", "test")
        }
        second = {
            split: trained.split_seed(42, split)
            for split in ("train", "validation", "test")
        }
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), 3)
        self.assertTrue(
            set(first.values()).isdisjoint(
                {
                    trained.split_seed(43, split)
                    for split in ("train", "validation", "test")
                }
            )
        )

    def test_identity_includes_target_and_entire_input(self):
        original = trained.example_identity((101, 200, 103, 102), 300)
        self.assertEqual(
            original,
            trained.example_identity((101, 200, 103, 102), 300),
        )
        self.assertNotEqual(
            original,
            trained.example_identity((101, 201, 103, 102), 300),
        )
        self.assertNotEqual(
            original,
            trained.example_identity((101, 200, 103, 102), 301),
        )


class TestCurriculum(unittest.TestCase):
    def test_every_complete_pass_contains_each_grid_cell_once(self):
        schedule = trained.curriculum_indices(108, 2160, seed=42)
        self.assertEqual(len(schedule), 2160)
        for start in range(0, 2160, 108):
            self.assertEqual(set(schedule[start : start + 108]), set(range(108)))
        self.assertEqual(schedule, trained.curriculum_indices(108, 2160, seed=42))
        self.assertNotEqual(schedule, trained.curriculum_indices(108, 2160, seed=43))

    def test_learning_rate_warms_up_and_decays(self):
        values = [
            trained.learning_rate_at_step(
                step,
                total_steps=2160,
                warmup_steps=108,
                peak_learning_rate=5e-5,
                minimum_learning_rate=5e-6,
            )
            for step in (0, 107, 108, 1000, 2159)
        ]
        self.assertLess(values[0], values[1])
        self.assertAlmostEqual(values[1], 5e-5)
        self.assertAlmostEqual(values[2], 5e-5)
        self.assertGreater(values[2], values[3])
        self.assertGreater(values[3], values[4])
        self.assertGreaterEqual(values[4], 5e-6)


class TestTrainedMQARJobs(unittest.TestCase):
    def test_dependency_gated_three_seed_jobs(self):
        job_root = NEOBERT_ROOT / "jobs" / "fineweb_evaluation"
        scripts = (
            job_root / "trained_masked_mqar.sbatch",
            job_root / "compare_trained_masked_mqar.sbatch",
            job_root / "submit_trained_masked_mqar.sh",
        )
        subprocess.run(["bash", "-n", *(str(path) for path in scripts)], check=True)
        training_text = scripts[0].read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gpus=A100:1", training_text)
        self.assertIn("--require-a100", training_text)
        self.assertIn("--steps 2", training_text)
        self.assertIn("MQAR_TRANSFER_STEPS:-2160", training_text)
        self.assertIn("--contexts 128 256 512 1024", training_text)
        launcher_text = scripts[2].read_text(encoding="utf-8")
        self.assertIn("TRANSFER_SEEDS=42", launcher_text)
        self.assertIn("TRANSFER_SEEDS=43,44", launcher_text)
        self.assertIn('dependency="afterok:$preflight_id"', launcher_text)
        self.assertIn('dependency="afterok:$pilot_id"', launcher_text)
        self.assertIn('dependency="afterok:$replication_id"', launcher_text)


if __name__ == "__main__":
    unittest.main()
