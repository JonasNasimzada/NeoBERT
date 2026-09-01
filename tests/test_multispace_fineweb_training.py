"""Contracts for the dedicated multispace FineWeb-Edu training workflow."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from omegaconf import OmegaConf


NEOBERT_ROOT = Path(__file__).resolve().parents[1]


class TestMultiSpaceFineWebTraining(unittest.TestCase):
    def test_dataset_contract_preserves_the_100m_model_and_token_budget(self):
        dataset = OmegaConf.load(
            NEOBERT_ROOT / "conf" / "dataset" / "fineweb_edu_google_1024.yaml"
        )
        tokenizer = OmegaConf.load(
            NEOBERT_ROOT / "conf" / "tokenizer" / "google-1024.yaml"
        )

        self.assertEqual(dataset.train.path, "HuggingFaceFW/fineweb-edu")
        self.assertEqual(dataset.train.name, "sample-10BT")
        self.assertEqual(
            dataset.train.revision,
            "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        )
        self.assertEqual(dataset.approx_token_limit, 1_600_000_000)
        self.assertEqual(dataset.pack_to_length, 1024)
        self.assertTrue(dataset.cross_document_attention)
        self.assertEqual(dataset.validation_fraction, 0.01)
        self.assertEqual(dataset.minimum_packed_rows, 1_344_000)
        self.assertEqual(tokenizer.vocab_size, 30_522)
        self.assertEqual(
            tokenizer.pretrained_model_name_or_path,
            "google-bert/bert-base-uncased",
        )

        schedule = dataset.training_schedule
        self.assertEqual(schedule.optimizer_steps, 84_000)
        self.assertEqual(schedule.global_sequences, 16)
        self.assertEqual(
            schedule.required_token_positions,
            schedule.optimizer_steps
            * schedule.global_sequences
            * dataset.pack_to_length,
        )
        self.assertEqual(
            dataset.minimum_packed_rows,
            schedule.optimizer_steps * schedule.global_sequences,
        )

    def test_jobs_are_syntax_valid_and_fixed_to_multispace_fineweb(self):
        jobs = NEOBERT_ROOT / "jobs" / "multispace_fineweb"
        prepare = (jobs / "prepare_data.sbatch").read_text(encoding="utf-8")
        train = (jobs / "train.sbatch").read_text(encoding="utf-8")
        submit = (jobs / "submit.sh").read_text(encoding="utf-8")

        for path in (
            jobs / "prepare_data.sbatch",
            jobs / "train.sbatch",
            jobs / "submit.sh",
        ):
            subprocess.run(["bash", "-n", str(path)], check=True)

        self.assertIn("dataset=fineweb_edu_google_1024", prepare)
        self.assertIn("tokenizer=google-1024", prepare)
        self.assertIn("#SBATCH --gpus=A100:1", train)
        self.assertIn("resolve_attention_variant 11", train)
        self.assertIn("dataset=fineweb_edu_google_1024", train)
        self.assertIn("tokenizer=google-1024", train)
        self.assertIn("model=attention-ablation-multispace", train)
        self.assertIn('MAX_STEPS="${MAX_STEPS:-84000}"', train)
        self.assertIn('TRAIN_SEGMENTS="${TRAIN_SEGMENTS:-5}"', submit)
        self.assertIn("afterok:", submit)
        self.assertNotIn("benchmark", submit)


if __name__ == "__main__":
    unittest.main()
