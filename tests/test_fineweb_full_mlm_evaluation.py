"""Contracts for the paired full FineWeb-Edu held-out MLM evaluation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
FULL_VALIDATION_TOKENS = 15_224_832
BATCH_TOKENS = 4_096
CONTEXTS = (128, 256, 512, 1024)


class TestFineWebFullMLMEvaluation(unittest.TestCase):
    def test_manifest_and_evaluation_geometry_cover_the_full_split(self):
        manifest_path = (
            NEOBERT_ROOT
            / "tokenized_datasets"
            / "fineweb_edu_google_1024_1p6b"
            / "optibertneo_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = manifest["splits"]["validation"]

        self.assertEqual(validation["tokens"], FULL_VALIDATION_TOKENS)
        self.assertEqual(validation["rows"] * 1024, FULL_VALIDATION_TOKENS)
        self.assertEqual(manifest["packing"]["cross_document_attention"], True)
        for context in CONTEXTS:
            self.assertEqual(FULL_VALIDATION_TOKENS % context, 0)
            self.assertEqual(BATCH_TOKENS % context, 0)

    def test_job_locks_a_fair_parameter_matched_pair(self):
        job = (
            NEOBERT_ROOT / "jobs" / "fineweb_evaluation" / "evaluate_mlm.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gpus=A100:1", job)
        self.assertIn("#SBATCH --array=0-1", job)
        self.assertIn('readonly FULL_VALIDATION_TOKENS=15224832', job)
        self.assertIn('readonly BATCH_TOKENS=4096', job)
        self.assertIn('--contexts 128 256 512 1024', job)
        self.assertIn('--token-budget "$FULL_VALIDATION_TOKENS"', job)
        self.assertIn('--batch-tokens "$BATCH_TOKENS"', job)
        self.assertIn('ATTENTION_VARIANT="multispace-flash"', job)
        self.assertIn('ATTENTION_VARIANT="real-100m-flash"', job)
        self.assertIn("fineweb-edu-s1024-multispace-100m-v1", job)
        self.assertIn("fineweb-edu-s1024-real-100m-v1", job)
        self.assertIn("--require-a100", job)

    def test_preflight_runs_both_exported_models_on_a100(self):
        job = (
            NEOBERT_ROOT / "jobs" / "fineweb_evaluation" / "preflight.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gpus=A100:1", job)
        self.assertIn("--variant multispace-flash", job)
        self.assertIn("--variant real-100m-flash", job)
        self.assertEqual(job.count("scripts/attention_ablation/benchmark_mlm.py"), 2)
        self.assertEqual(job.count("--token-budget 4096"), 2)


if __name__ == "__main__":
    unittest.main()
