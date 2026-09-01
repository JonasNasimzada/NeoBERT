"""Determinism, pairing, aggregation, and job-contract tests for masked MQAR."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import subprocess
import sys
import unittest
from fractions import Fraction
from pathlib import Path


NEOBERT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = NEOBERT_ROOT / "scripts" / "attention_ablation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_for_tests", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


masked_mqar = load_script("masked_mqar")
compare_masked_mqar = load_script("compare_masked_mqar")


def alphabetic_symbol(index: int) -> str:
    letters = []
    value = index
    while True:
        letters.append(chr(ord("a") + value % 26))
        value = value // 26
        if value == 0:
            break
    return "z" + "".join(reversed(letters))


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103
    unk_token_id = 100
    all_special_ids = [0, 100, 101, 102, 103]

    def __init__(self):
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 100,
            "[CLS]": 101,
            "[SEP]": 102,
            "[MASK]": 103,
            ".": 119,
            ":": 131,
            ";": 132,
        }
        self.vocab.update(
            {alphabetic_symbol(index): 200 + index for index in range(512)}
        )

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def get_vocab(self):
        return dict(self.vocab)


class TestMaskedMQARGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FakeTokenizer()
        cls.markers = masked_mqar.resolve_marker_tokens(cls.tokenizer)
        cls.pool = masked_mqar.build_candidate_token_ids(cls.tokenizer, cls.markers)

    def test_default_grid_is_complete_factorial(self):
        grid = masked_mqar.make_grid(
            masked_mqar.DEFAULT_CONTEXTS,
            masked_mqar.DEFAULT_BINDINGS,
            masked_mqar.DEFAULT_DISTRACTORS,
            masked_mqar.DEFAULT_DISTANCE_FRACTIONS,
        )
        self.assertEqual(len(grid), 4 * 3 * 3 * 3)
        self.assertEqual({cell.context_length for cell in grid}, {128, 256, 512, 1024})
        self.assertEqual({cell.difficulty for cell in grid}, {"easy", "medium", "hard"})
        for cell in grid:
            expected = (
                cell.context_length
                * cell.distance_numerator
                // cell.distance_denominator
            )
            self.assertEqual(cell.query_distance, expected)

    def test_examples_have_exact_distance_and_disjoint_symbol_sets(self):
        cells = masked_mqar.make_grid(
            (128, 1024),
            (4, 16),
            (0, 32),
            (Fraction(1, 4), Fraction(3, 4)),
        )
        for cell in cells:
            for example_index in range(3):
                example = masked_mqar.generate_example(
                    cell,
                    example_index,
                    seed=77,
                    candidate_token_ids=self.pool,
                    markers=self.markers,
                )
                self.assertEqual(len(example.input_ids), cell.context_length)
                self.assertEqual(example.input_ids[0], self.markers.cls)
                self.assertEqual(example.input_ids[-1], self.markers.sep)
                self.assertEqual(example.input_ids[-2], self.markers.mask)
                self.assertEqual(example.input_ids[-3], example.query_key_token_id)
                self.assertEqual(example.input_ids.count(self.markers.mask), 1)
                self.assertEqual(
                    example.mask_position - example.target_value_position,
                    cell.query_distance,
                )
                self.assertEqual(
                    example.input_ids[example.target_value_position],
                    example.target_token_id,
                )
                self.assertEqual(
                    example.input_ids[example.target_value_position - 1],
                    self.markers.relation,
                )
                self.assertEqual(set(example.keys) & set(example.values), set())
                self.assertEqual(set(example.keys) & set(example.distractors), set())
                self.assertEqual(set(example.values) & set(example.distractors), set())

    def test_generation_and_fingerprint_are_reproducible(self):
        cell = masked_mqar.make_grid(
            (128,), (8,), (16,), (Fraction(1, 2),)
        )[0]

        def generate(seed):
            digest = hashlib.sha256()
            inputs, labels = masked_mqar.generate_cell_tensor(
                cell,
                examples_per_cell=16,
                seed=seed,
                candidate_token_ids=self.pool,
                markers=self.markers,
                fingerprint=digest,
            )
            return inputs, labels, digest.hexdigest()

        first = generate(123)
        second = generate(123)
        changed = generate(124)
        self.assertTrue(first[0].equal(second[0]))
        self.assertTrue(first[1].equal(second[1]))
        self.assertEqual(first[2], second[2])
        self.assertNotEqual(first[2], changed[2])


class TestMaskedMQARSummaries(unittest.TestCase):
    def test_micro_and_macro_metrics_are_both_explicit(self):
        reports = [
            {
                "examples": 2,
                "correct": 1,
                "nll_sum": 4.0,
                "accuracy": 0.5,
                "masked_nll": 2.0,
                "elapsed_seconds": 1.0,
                "evaluated_token_positions": 256,
            },
            {
                "examples": 4,
                "correct": 3,
                "nll_sum": 16.0,
                "accuracy": 0.75,
                "masked_nll": 4.0,
                "elapsed_seconds": 3.0,
                "evaluated_token_positions": 1024,
            },
        ]
        summary = masked_mqar.summarize_cells(reports)
        self.assertEqual(summary["examples"], 6)
        self.assertAlmostEqual(summary["micro_accuracy"], 4 / 6)
        self.assertAlmostEqual(summary["micro_masked_nll"], 20 / 6)
        self.assertAlmostEqual(summary["macro_cell_accuracy"], 0.625)
        self.assertAlmostEqual(summary["token_positions_per_second"], 320.0)


def result_summary(accuracy, nll, throughput):
    return {
        "examples": 2,
        "micro_accuracy": accuracy,
        "micro_masked_nll": nll,
        "token_positions_per_second": throughput,
    }


def fake_report(variant, outcomes, nll, throughput):
    accuracy = sum(outcomes) / len(outcomes)
    summary = result_summary(accuracy, sum(nll) / len(nll), throughput)
    groups = {
        "by_context_length": {"128": copy.deepcopy(summary)},
        "by_binding_count": {"4": copy.deepcopy(summary)},
        "by_distractor_count": {"0": copy.deepcopy(summary)},
        "by_query_distance_fraction": {"1/4": copy.deepcopy(summary)},
        "by_difficulty": {"easy": copy.deepcopy(summary)},
    }
    return {
        "benchmark": "masked-mqar",
        "model_path": f"/{variant}",
        "model": {
            "variant": variant,
            "trainable_parameters": 99_985_152,
        },
        "training_completion": {"completed_schedule": True},
        "runtime": {
            "device_name": "NVIDIA A100-SXM4-80GB",
            "autocast_dtype": "torch.bfloat16",
        },
        "protocol": {"version": "masked-mqar-v1", "dataset_sha256": "same"},
        "cells": {
            "c128_b4_d0_q25": {
                "context_length": 128,
                "binding_count": 4,
                "distractor_count": 0,
                "query_distance": 32,
                "query_distance_fraction": "1/4",
                "difficulty": "easy",
                "examples": 2,
                "accuracy": accuracy,
                "masked_nll": sum(nll) / len(nll),
                "token_positions_per_second": throughput,
                "example_correct": outcomes,
                "example_nll": nll,
            }
        },
        "summaries": {"overall": copy.deepcopy(summary), **groups},
    }


class TestPairedComparison(unittest.TestCase):
    def test_comparison_uses_paired_item_outcomes(self):
        multispace = fake_report("multispace-flash", [True, False], [1.0, 3.0], 100.0)
        real = fake_report("real-flash", [False, False], [2.0, 2.0], 200.0)
        comparison = compare_masked_mqar.compare_reports(multispace, real)
        self.assertTrue(comparison["parameter_matched"])
        self.assertEqual(comparison["dataset_sha256"], "same")
        self.assertAlmostEqual(
            comparison["overall"]["accuracy_difference_multispace_minus_real"],
            0.5,
        )
        self.assertAlmostEqual(
            comparison["overall"]["real_over_multispace_throughput_ratio"],
            2.0,
        )
        self.assertEqual(
            comparison["overall"]["paired_outcomes"]["multispace_only_correct"],
            1,
        )
        self.assertEqual(
            comparison["overall"]["paired_accuracy_difference_ci"]["paired_examples"],
            2,
        )

    def test_comparison_rejects_nonidentical_generated_data(self):
        multispace = fake_report("multispace-flash", [True, False], [1.0, 3.0], 100.0)
        real = fake_report("real-flash", [False, False], [2.0, 2.0], 200.0)
        real["protocol"]["dataset_sha256"] = "different"
        with self.assertRaisesRegex(AssertionError, "protocol manifests"):
            compare_masked_mqar.compare_reports(multispace, real)


class TestMaskedMQARJobs(unittest.TestCase):
    def test_jobs_are_valid_shell_and_enforce_a100_pair(self):
        job_root = NEOBERT_ROOT / "jobs" / "fineweb_evaluation"
        scripts = (
            job_root / "masked_mqar.sbatch",
            job_root / "compare_masked_mqar.sbatch",
            job_root / "submit_masked_mqar.sh",
        )
        subprocess.run(["bash", "-n", *(str(path) for path in scripts)], check=True)
        evaluation_text = scripts[0].read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gpus=A100:1", evaluation_text)
        self.assertIn("--require-a100", evaluation_text)
        self.assertIn('ATTENTION_VARIANT="multispace-flash"', evaluation_text)
        self.assertIn('ATTENTION_VARIANT="real-flash"', evaluation_text)
        self.assertIn("--contexts 128 256 512 1024", evaluation_text)
        self.assertIn("--bindings 4 8 16", evaluation_text)
        self.assertIn("--distractors 0 16 32", evaluation_text)
        submit_text = scripts[2].read_text(encoding="utf-8")
        self.assertIn("afterok:$smoke_job_id", submit_text)
        self.assertIn("afterok:$full_job_id", submit_text)


if __name__ == "__main__":
    unittest.main()
