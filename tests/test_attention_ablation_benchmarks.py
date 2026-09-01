"""CPU-only tests for attention-ablation benchmark helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


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


benchmark_mlm = load_script("benchmark_mlm")
log_babylm_results = load_script("log_babylm_results")


def load_flash_adapter():
    path = (
        NEOBERT_ROOT
        / "scripts"
        / "attention_ablation"
        / "babylm_compat"
        / "padding_free_flash.py"
    )
    spec = importlib.util.spec_from_file_location(
        "padding_free_flash_for_tests",
        path,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


padding_free_flash = load_flash_adapter()


class TestVariantMatrix(unittest.TestCase):
    def test_flash_variants_are_canonicalized_and_registered(self):
        self.assertEqual(benchmark_mlm.canonical_variant("real_torch"), "real-torch")
        self.assertEqual(benchmark_mlm.canonical_variant("REAL-FLASH"), "real-flash")
        self.assertEqual(
            benchmark_mlm.VARIANT_MATRIX["real-torch"],
            ("real", "torch"),
        )
        self.assertEqual(
            benchmark_mlm.VARIANT_MATRIX["real-flash"],
            ("real", "flash"),
        )
        self.assertEqual(
            benchmark_mlm.canonical_variant("REAL_100M_FLASH"),
            "real-100m-flash",
        )
        self.assertEqual(
            benchmark_mlm.VARIANT_MATRIX["real-100m-flash"],
            ("real", "flash"),
        )
        self.assertEqual(
            benchmark_mlm.canonical_variant("SPLIT_FLASH"),
            "split-flash",
        )
        self.assertEqual(
            benchmark_mlm.VARIANT_MATRIX["split-flash"],
            ("split", "flash"),
        )
        self.assertEqual(
            benchmark_mlm.canonical_variant("DUAL_FLASH"),
            "dual-flash",
        )
        self.assertEqual(
            benchmark_mlm.VARIANT_MATRIX["dual-flash"],
            ("dual", "flash"),
        )
        self.assertEqual(
            benchmark_mlm.canonical_variant("MULTISPACE_FLASH"),
            "multispace-flash",
        )
        self.assertEqual(
            benchmark_mlm.VARIANT_MATRIX["multispace-flash"],
            ("multispace", "flash"),
        )
        self.assertEqual(
            benchmark_mlm.expected_variant_schedule("multispace-flash", 9),
            (
                ("multispace",) * 9,
                ("flash",) * 9,
            ),
        )
        self.assertEqual(
            benchmark_mlm.expected_variant_schedule("multispace-flash", 3),
            (("multispace",) * 3, ("flash",) * 3),
        )
        self.assertEqual(
            benchmark_mlm.expected_trainable_parameters("complex-native"),
            17_260_288,
        )
        self.assertEqual(
            benchmark_mlm.expected_trainable_parameters("multispace-flash"),
            99_985_152,
        )
        self.assertEqual(
            benchmark_mlm.expected_trainable_parameters("real-flash"),
            17_260_288,
        )
        self.assertEqual(
            benchmark_mlm.expected_trainable_parameters("real-100m-flash"),
            99_985_152,
        )
        with self.assertRaisesRegex(
            AssertionError,
            "checkpoint uses attention spaces",
        ):
            benchmark_mlm.validate_variant_schedule(
                "multispace-flash",
                (
                    "split",
                    "multispace",
                    "multispace",
                    "multispace",
                    "multispace",
                    "multispace",
                    "multispace",
                    "multispace",
                    "multispace",
                ),
                ("flash",) * 9,
            )
        self.assertEqual(
            set(benchmark_mlm.VARIANT_MATRIX),
            {
                "complex-native",
                "complex-torch",
                "complex-flash",
                "split-native",
                "split-torch",
                "real-torch",
                "real-flash",
                "real-100m-flash",
                "split-flash",
                "dual-native",
                "dual-torch",
                "dual-flash",
                "multispace-flash",
            },
        )

    def test_heldout_token_budget_fits_prepared_validation_split(self):
        self.assertEqual(
            benchmark_mlm.DEFAULT_CONTEXT_LENGTHS,
            (128, 256, 512, 1024),
        )
        self.assertEqual(benchmark_mlm.DEFAULT_TOKEN_BUDGET, 1_732_608)
        self.assertEqual(
            benchmark_mlm.DEFAULT_TOKEN_BUDGET
            % benchmark_mlm.DEFAULT_BATCH_TOKENS,
            0,
        )
        for context_length in benchmark_mlm.DEFAULT_CONTEXT_LENGTHS:
            self.assertEqual(
                benchmark_mlm.DEFAULT_TOKEN_BUDGET % context_length,
                0,
            )


class TestBenchmarkMemoryMetrics(unittest.TestCase):
    def test_allocator_peaks_include_baseline_and_incremental_workspace(self):
        properties = types.SimpleNamespace(total_memory=80 * 2**30)
        with (
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=900),
            mock.patch.object(torch.cuda, "max_memory_reserved", return_value=1200),
            mock.patch.object(torch.cuda, "get_device_properties", return_value=properties),
        ):
            metrics = benchmark_mlm.cuda_memory_metrics(
                torch.device("cuda"),
                baseline_allocated_bytes=600,
                baseline_reserved_bytes=1000,
            )

        self.assertEqual(metrics["peak_cuda_workspace_allocated_bytes"], 300)
        self.assertEqual(metrics["peak_cuda_workspace_reserved_bytes"], 200)
        self.assertEqual(metrics["peak_memory_bytes"], 900)
        self.assertEqual(metrics["peak_memory_reserved_bytes"], 1200)
        self.assertEqual(metrics["device_total_memory_bytes"], 80 * 2**30)


class TestMLMQualityMetrics(unittest.TestCase):
    def test_quality_aggregate_is_weighted_by_masked_tokens(self):
        aggregate = benchmark_mlm.aggregate_quality_metrics(
            {
                "128": {
                    "masked_tokens": 2,
                    "masked_token_correct": 1,
                    "mlm_cross_entropy_loss": 2.0,
                },
                "1024": {
                    "masked_tokens": 3,
                    "masked_token_correct": 2,
                    "mlm_cross_entropy_loss": 4.0,
                },
            }
        )

        self.assertEqual(aggregate["masked_token_evaluations"], 5)
        self.assertEqual(aggregate["mlm_cross_entropy_loss"], 3.2)
        self.assertEqual(aggregate["masked_token_top1_accuracy"], 0.6)


class ListDataset:
    column_names = ["input_ids"]

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class TestDeterministicMasking(unittest.TestCase):
    def test_masking_is_reproducible_and_never_masks_special_tokens(self):
        input_ids = torch.arange(256, dtype=torch.long).view(2, 128)
        special_ids = {0, 101, 102}

        def mask_once():
            generator = torch.Generator(device="cpu").manual_seed(17)
            return benchmark_mlm.deterministic_mask_inputs(
                input_ids,
                mask_token_id=103,
                special_token_ids=special_ids,
                probability=0.20,
                generator=generator,
            )

        first = mask_once()
        second = mask_once()
        for actual, expected in zip(first, second):
            torch.testing.assert_close(actual, expected)

        corrupted, labels, eligible = first
        for special_id in special_ids:
            positions = input_ids.eq(special_id)
            self.assertTrue(labels[positions].eq(-100).all())
            self.assertTrue(corrupted[positions].eq(input_ids[positions]).all())
            self.assertTrue(eligible[positions].logical_not().all())
        self.assertGreater(labels.ne(-100).sum().item(), 0)

    def test_random_stream_is_invariant_to_context_reshaping(self):
        input_ids = torch.arange(512, dtype=torch.long)

        def mask_in_shape(shape):
            generator = torch.Generator(device="cpu").manual_seed(123)
            _, labels, _ = benchmark_mlm.deterministic_mask_inputs(
                input_ids.view(shape),
                mask_token_id=103,
                special_token_ids=(),
                probability=0.20,
                generator=generator,
            )
            return labels.reshape(-1)

        torch.testing.assert_close(mask_in_shape((4, 128)), mask_in_shape((2, 256)))

    def test_equal_token_prefix_is_chunked_at_each_context(self):
        rows = [
            {"input_ids": list(range(256))},
            {"input_ids": list(range(256, 512))},
        ]
        dataset = ListDataset(rows)

        flattened = {}
        for context in (128, 256, 512):
            batches = benchmark_mlm.iter_fixed_token_batches(
                dataset,
                context_length=context,
                token_budget=512,
                batch_tokens=512,
            )
            flattened[context] = torch.cat(
                [batch["input_ids"].reshape(-1) for batch in batches]
            )

        torch.testing.assert_close(flattened[128], flattened[256])
        torch.testing.assert_close(flattened[128], flattened[512])


class TestTrainingCompletionGate(unittest.TestCase):
    def test_completion_flag_and_legacy_step_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory)
            summary_path = model_directory / "training_summary.json"
            missing_directory = model_directory / "missing"
            missing_directory.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "training summary is missing"):
                benchmark_mlm.validate_training_completion(missing_directory)
            missing_status = benchmark_mlm.validate_training_completion(
                missing_directory,
                allow_incomplete=True,
            )
            self.assertEqual(missing_status["status"], "allowed_missing")

            summary_path.write_text(
                json.dumps(
                    {
                        "optimizer_steps": 10,
                        "train/completed_schedule": True,
                        "resolved_config": {"trainer": {"max_steps": 10}},
                    }
                ),
                encoding="utf-8",
            )
            status = benchmark_mlm.validate_training_completion(model_directory)
            self.assertEqual(status["status"], "complete")

            summary_path.write_text(
                json.dumps(
                    {
                        "optimizer_steps": 10,
                        "resolved_config": {"trainer": {"max_steps": 10}},
                    }
                ),
                encoding="utf-8",
            )
            status = benchmark_mlm.validate_training_completion(model_directory)
            self.assertEqual(status["status"], "complete")

            summary_path.write_text(
                json.dumps(
                    {
                        "optimizer_steps": 9,
                        "train/completed_schedule": False,
                        "resolved_config": {"trainer": {"max_steps": 10}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                benchmark_mlm.validate_training_completion(model_directory)
            status = benchmark_mlm.validate_training_completion(
                model_directory,
                allow_incomplete=True,
            )
            self.assertEqual(status["status"], "allowed_incomplete")


class TestBabyLMResultParsing(unittest.TestCase):
    def test_recursive_json_and_key_value_reports_are_namespaced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "zeroshot").mkdir()
            (root / "zeroshot" / "scores.json").write_text(
                json.dumps(
                    {
                        "blimp": {"accuracy": 0.75, "passed": True},
                        "tasks": [{"score": 0.5}, {"score": "ignored"}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "finetune.txt").write_text(
                "glue/mnli: 0.812\n"
                "macro average: 72.5%\n"
                "notes: ignored\n"
                "### AVERAGE ACCURACY\n"
                "0.667\n",
                encoding="utf-8",
            )

            metrics, sources = log_babylm_results.collect_metrics(root)

        self.assertEqual(
            metrics["benchmark/babylm/zeroshot/scores/blimp/accuracy"],
            0.75,
        )
        self.assertEqual(
            metrics["benchmark/babylm/zeroshot/scores/tasks/0/score"],
            0.5,
        )
        self.assertEqual(metrics["benchmark/babylm/finetune/glue/mnli"], 0.812)
        self.assertEqual(metrics["benchmark/babylm/finetune/macro_average"], 0.725)
        self.assertEqual(
            metrics["benchmark/babylm/finetune/average_accuracy"],
            0.667,
        )
        self.assertNotIn(
            "benchmark/babylm/zeroshot/scores/blimp/passed",
            metrics,
        )
        self.assertEqual(
            sources["benchmark/babylm/finetune/glue/mnli"],
            "finetune.txt",
        )

    def test_report_sections_disambiguate_and_normalize_official_accuracy(self):
        parsed = log_babylm_results.parse_key_value_report(
            "### FIELD ACCURACY\n"
            "supplement: 62.50\n"
            "### LINGUISTICS TERM ACCURACY\n"
            "supplement: 75.00\n"
            "### AVERAGE ACCURACY\n"
            "68.75\n"
        )

        self.assertEqual(parsed[("field_accuracy", "supplement")], 0.625)
        self.assertEqual(
            parsed[("linguistics_term_accuracy", "supplement")],
            0.75,
        )
        self.assertEqual(parsed[("average_accuracy",)], 0.6875)

    def test_distinct_report_labels_survive_normalization_collisions(self):
        parsed = log_babylm_results.parse_key_value_report(
            "### CONTEXT CONTRAST ACCURACY\n"
            "variable swap: 49.77\n"
            "variable_swap: 63.33\n"
        )

        self.assertAlmostEqual(
            parsed[("context_contrast_accuracy", "variable_swap")],
            0.4977,
        )
        self.assertAlmostEqual(
            parsed[("context_contrast_accuracy", "variable_swap_2")],
            0.6333,
        )

    def test_genuinely_repeated_report_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate report key"):
            log_babylm_results.parse_key_value_report(
                "### UID ACCURACY\n"
                "same key: 50.00\n"
                "same key: 51.00\n"
            )

    def test_resource_report_is_logged_under_system_namespace(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "resources.json"
            report.write_text(
                json.dumps(
                    {
                        "peak_gpu_process_memory_bytes": 12_345,
                        "peak_host_rss_bytes": 67_890,
                        "nested": {"samples": 7},
                        "available": True,
                    }
                ),
                encoding="utf-8",
            )
            metrics, sources = log_babylm_results.collect_resource_metrics(report)

        self.assertEqual(
            metrics["benchmark/babylm/system/peak_gpu_process_memory_bytes"],
            12_345,
        )
        self.assertEqual(
            metrics["benchmark/babylm/system/nested/samples"],
            7,
        )
        self.assertNotIn("benchmark/babylm/system/available", metrics)
        self.assertEqual(
            sources["benchmark/babylm/system/peak_host_rss_bytes"],
            str(report.resolve()),
        )

    def test_offline_wandb_jobs_use_stable_resumable_ids(self):
        init_calls = []

        class FakeRun:
            id = "fake-run"

            def log(self, values):
                self.logged = values

            def log_artifact(self, artifact):
                self.artifact = artifact

            def finish(self):
                self.finished = True

        class FakeArtifact:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def add_file(self, *args, **kwargs):
                pass

            def add_dir(self, *args, **kwargs):
                pass

        def fake_init(**kwargs):
            init_calls.append(kwargs)
            return FakeRun()

        fake_wandb = types.SimpleNamespace(
            init=fake_init,
            Artifact=FakeArtifact,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "results.json"
            output.write_text("{}\n", encoding="utf-8")
            common = {
                "mode": "offline",
                "variant": "complex-native",
                "seed": 42,
                "project": "test-project",
                "entity": "",
                "name": None,
                "group": None,
                "id_prefix": "a100-3h-v1",
            }
            benchmark_args = types.SimpleNamespace(
                **common,
                model=root / "final_model",
                dataset=root / "dataset",
                split="validation",
                contexts=(128, 256, 512),
                token_budget=512,
                batch_tokens=512,
                mask_probability=0.20,
                allow_incomplete=False,
            )
            benchmark_report = {
                "trainable_parameters": 17_260_288,
                "training_completion": {"status": "complete"},
                "quality_aggregate": {},
                "results": {},
            }
            babylm_args = types.SimpleNamespace(
                **common,
                results=root,
            )

            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                benchmark_mlm._log_to_wandb(
                    benchmark_report,
                    output,
                    benchmark_args,
                )
                log_babylm_results._log_to_wandb(
                    {"benchmark/babylm/average_accuracy": 0.75},
                    output,
                    babylm_args,
                )

        self.assertEqual(
            [call["id"] for call in init_calls],
            [
                "a100-3h-v1-complex-native-seed-42-heldout-mlm",
                "a100-3h-v1-complex-native-seed-42-babylm",
            ],
        )
        self.assertEqual([call["resume"] for call in init_calls], ["allow", "allow"])
        self.assertEqual([call["mode"] for call in init_calls], ["offline", "offline"])


class TestBabyLMFlashCompatibility(unittest.TestCase):
    def test_padded_batch_is_evaluated_in_padding_free_length_groups(self):
        class FakeFlashModel(torch.nn.Module):
            def __init__(self, backend):
                super().__init__()
                self.config = types.SimpleNamespace(attention_backends=[backend])
                self.calls = []

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                del kwargs
                self.calls.append(
                    (
                        tuple(input_ids.shape),
                        None if attention_mask is None else attention_mask.clone(),
                    )
                )
                logits = input_ids.unsqueeze(-1).expand(*input_ids.shape, 5).float()
                return types.SimpleNamespace(logits=logits)

        input_ids = torch.tensor(
            [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 0]]
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
        )

        for backend in ("flash", "flash_fused"):
            with self.subTest(backend=backend):
                model = padding_free_flash.install_padding_free_flash_forward(
                    FakeFlashModel(backend)
                )
                output = model(input_ids=input_ids, attention_mask=attention_mask)

                self.assertEqual(tuple(output.logits.shape), (3, 4, 5))
                self.assertEqual(
                    [shape for shape, _ in model.calls],
                    [(2, 3), (1, 2)],
                )
                self.assertTrue(all(mask is None for _, mask in model.calls))
                torch.testing.assert_close(
                    output.logits[0, :3, 0], input_ids[0, :3].float()
                )
                torch.testing.assert_close(
                    output.logits[1, :2, 0], input_ids[1, :2].float()
                )
                torch.testing.assert_close(
                    output.logits[2, :3, 0], input_ids[2, :3].float()
                )
                self.assertTrue(output.logits[0, 3].eq(0).all())
                self.assertTrue(output.logits[1, 2:].eq(0).all())
                self.assertTrue(output.logits[2, 3].eq(0).all())

                model.calls.clear()
                model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
                self.assertEqual(model.calls, [((3, 4), None)])


if __name__ == "__main__":
    unittest.main()
