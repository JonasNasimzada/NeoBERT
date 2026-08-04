"""CPU-only tests for the OptiBERTneo training preflight."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    NEOBERT_ROOT
    / "scripts"
    / "pretraining"
    / "preflight_optibertneo.py"
)
SPEC = importlib.util.spec_from_file_location(
    "optibertneo_preflight_for_tests",
    PREFLIGHT_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {PREFLIGHT_PATH}")
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


def failures(checks):
    return [check for check in checks if check.failed]


class TestRecipeAndModelMath(unittest.TestCase):
    def test_two_node_recipe_is_exactly_the_1p3b_token_schedule(self):
        recipe = preflight.Recipe(
            nodes=2,
            gpus_per_node=4,
            micro_batch=32,
        )

        self.assertEqual(recipe.world_size, 8)
        self.assertEqual(recipe.gradient_accumulation_steps, 8)
        self.assertEqual(recipe.tokens_per_optimizer_step, 2_097_152)
        self.assertEqual(recipe.scheduled_rows, 1_269_760)
        self.assertEqual(recipe.scheduled_tokens, 1_300_234_240)
        self.assertEqual(
            recipe.scheduled_rows,
            preflight.MINIMUM_PACKED_ROWS,
        )

    def test_recipe_rejects_fractional_gradient_accumulation(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            preflight.Recipe(
                nodes=2,
                gpus_per_node=4,
                micro_batch=30,
            )

    def test_report_returns_nonzero_for_an_actual_failure(self):
        report = preflight.Report()
        report.failed("example", "expected test failure")

        self.assertEqual(report.exit_code, 1)

    def test_real_model_count_matches_shipped_inspector(self):
        counts = preflight.calculate_real_model_counts()

        self.assertEqual(counts.paper_target, 198_180_864)
        self.assertEqual(counts.non_embedding, 198_225_408)
        self.assertEqual(counts.total, 236_828_928)
        self.assertLess(abs(counts.relative_paper_difference), 0.001)

    def test_shipped_configuration_and_launcher_are_consistent(self):
        checks = preflight._configuration_checks(NEOBERT_ROOT)

        self.assertEqual(
            failures(checks),
            [],
            "\n".join(f"{check.name}: {check.detail}" for check in checks),
        )

    def test_config_only_cli_needs_no_training_imports(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = preflight.main(
                ["--config-only", "--project-root", str(NEOBERT_ROOT)]
            )

        self.assertEqual(exit_code, 0, output.getvalue())
        self.assertIn("1,300,234,240 scheduled tokens", output.getvalue())
        self.assertIn("[SKIP] runtime", output.getvalue())

    def test_slurm_is_not_inferred_for_dataset_only_runs(self):
        environment = {"SLURM_JOB_ID": "123", "SLURM_JOB_NUM_NODES": "1"}

        self.assertFalse(
            preflight.should_validate_slurm(
                check_slurm=False,
                skip_slurm=False,
                require_gpu=False,
                environment=environment,
            )
        )
        self.assertTrue(
            preflight.should_validate_slurm(
                check_slurm=False,
                skip_slurm=False,
                require_gpu=True,
                environment=environment,
            )
        )


class TestDatasetValidation(unittest.TestCase):
    def valid_summary(self):
        return preflight.DatasetSummary(
            column_names=("input_ids", "document_ids"),
            num_rows=1_269_760,
            min_lengths={"input_ids": 1024, "document_ids": 1024},
            max_lengths={"input_ids": 1024, "document_ids": 1024},
            element_types={
                "input_ids": "int32",
                "document_ids": "int32",
            },
            null_counts={"input_ids": 0, "document_ids": 0},
            min_values={"document_ids": 0},
            fingerprint="synthetic-fingerprint",
        )

    def valid_manifest(self):
        return {
            "format_version": 1,
            "source": dict(preflight.DATASET_SOURCE),
            "source_token_limit": 1_600_000_000,
            "dataset_fingerprint": "synthetic-fingerprint",
            "rows": 1_269_760,
            "sequence_length": 1024,
            "packed_token_positions": 1_300_234_240,
            "packing": {
                "padding_free": True,
                "cross_document_attention": False,
                "document_id_padding_value": None,
            },
            "tokenizer": dict(preflight.TOKENIZER_IDENTITY),
            "paper_schedule": {
                "optimizer_steps": 620,
                "global_sequences": 2048,
                "required_token_positions": 1_300_234_240,
            },
        }

    def test_valid_packed_dataset_summary_passes(self):
        checks = preflight.validate_dataset_summary(self.valid_summary())

        self.assertEqual(failures(checks), [])

    def test_valid_manifest_matches_arrow_summary_and_pins(self):
        checks = preflight.validate_optibertneo_manifest(
            self.valid_manifest(),
            self.valid_summary(),
        )

        self.assertEqual(failures(checks), [])

    def test_manifest_rejects_padding_wrong_counts_and_unpinned_inputs(self):
        manifest = self.valid_manifest()
        manifest["rows"] -= 1
        manifest["packing"]["padding_free"] = False
        manifest["source"]["revision"] = "main"
        manifest["tokenizer"]["mask_token_id"] = 3
        manifest["paper_schedule"]["optimizer_steps"] = 619

        checks = preflight.validate_optibertneo_manifest(
            manifest,
            self.valid_summary(),
        )
        failed_names = {check.name for check in failures(checks)}

        self.assertIn("dataset.manifest.counts", failed_names)
        self.assertIn("dataset.manifest.packing", failed_names)
        self.assertIn("dataset.manifest.source", failed_names)
        self.assertIn("dataset.manifest.tokenizer", failed_names)
        self.assertIn("dataset.manifest.schedule", failed_names)

    def test_negative_document_id_rejects_padded_dataset(self):
        summary = self.valid_summary()
        padded_summary = preflight.DatasetSummary(
            column_names=summary.column_names,
            num_rows=summary.num_rows,
            min_lengths=summary.min_lengths,
            max_lengths=summary.max_lengths,
            element_types=summary.element_types,
            null_counts=summary.null_counts,
            min_values={"document_ids": -1},
            fingerprint=summary.fingerprint,
        )

        checks = preflight.validate_dataset_summary(padded_summary)

        self.assertIn(
            "dataset.document_ids.padding",
            {check.name for check in failures(checks)},
        )

    def test_missing_column_short_row_and_too_few_rows_fail(self):
        summary = preflight.DatasetSummary(
            column_names=("input_ids",),
            num_rows=1_269_759,
            min_lengths={"input_ids": 1023},
            max_lengths={"input_ids": 1024},
            element_types={"input_ids": "int64"},
            null_counts={"input_ids": 1},
            min_values={},
        )

        checks = preflight.validate_dataset_summary(summary)
        failed_names = {check.name for check in failures(checks)}

        self.assertIn("dataset.schema", failed_names)
        self.assertIn("dataset.rows", failed_names)
        self.assertIn("dataset.input_ids.length", failed_names)
        self.assertIn("dataset.input_ids.dtype", failed_names)
        self.assertIn("dataset.input_ids.nulls", failed_names)

    def test_dataset_path_failure_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            checks = preflight._dataset_checks(missing, datasets_module=None)

        self.assertTrue(failures(checks))
        self.assertEqual(failures(checks)[0].name, "dataset.path")


class TestEnvironmentValidators(unittest.TestCase):
    def test_deepspeed_import_uses_temporary_read_only_cache_override(self):
        requirement = preflight.ImportRequirement(
            "deepspeed",
            "deepspeed",
            (0, 15, 4),
            (0, 15, 5),
        )
        observed_cache_directories = []

        def fake_import(module_name):
            self.assertEqual(module_name, "deepspeed")
            observed_cache_directories.append(
                os.environ.get("TRITON_CACHE_DIR")
            )
            return object()

        original_cache = os.environ.pop("TRITON_CACHE_DIR", None)
        try:
            with (
                mock.patch.object(
                    preflight,
                    "IMPORT_REQUIREMENTS",
                    (requirement,),
                ),
                mock.patch.object(
                    preflight.importlib,
                    "import_module",
                    side_effect=fake_import,
                ),
                mock.patch.object(
                    preflight,
                    "_module_version",
                    return_value="0.15.4",
                ),
            ):
                checks, modules = preflight._import_checks()
        finally:
            if original_cache is not None:
                os.environ["TRITON_CACHE_DIR"] = original_cache

        self.assertEqual(failures(checks), [])
        self.assertIn("deepspeed", modules)
        self.assertEqual(observed_cache_directories, ["/tmp"])
        if original_cache is None:
            self.assertNotIn("TRITON_CACHE_DIR", os.environ)
        else:
            self.assertEqual(
                os.environ.get("TRITON_CACHE_DIR"),
                original_cache,
            )

    def test_valid_slurm_environment_is_two_by_four(self):
        environment = {
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_GPUS_ON_NODE": "gpu:h100:4",
            "WORLD_SIZE": "8",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "SLURM_NODEID": "1",
        }

        checks = preflight.validate_slurm_environment(environment)

        self.assertEqual(failures(checks), [])
        self.assertEqual(preflight.parse_slurm_gpu_count("4(S:0-3)"), 4)

    def test_wrong_slurm_shape_fails_nodes_gpus_and_world(self):
        environment = {
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_GPUS_ON_NODE": "8",
            "WORLD_SIZE": "16",
        }

        checks = preflight.validate_slurm_environment(environment)
        failed_names = {check.name for check in failures(checks)}

        self.assertIn("slurm.nodes", failed_names)
        self.assertIn("slurm.gpus_per_node", failed_names)
        self.assertIn("slurm.world_size", failed_names)

    def test_sm90_architecture_spellings(self):
        for value in (
            "sm_90",
            "compute_90",
            "9.0",
            "9.0a",
            "8.0;8.6;9.0+PTX",
        ):
            with self.subTest(value=value):
                self.assertTrue(preflight.supports_sm90(value))
        self.assertFalse(preflight.supports_sm90("8.0;8.6+PTX"))

    def test_cuda_build_version_encodings(self):
        self.assertEqual(preflight._cuda_version_tuple(1201), (12, 1))
        self.assertEqual(preflight._cuda_version_tuple(12060), (12, 6))
        self.assertEqual(preflight._cuda_version_tuple("12.6"), (12, 6))

    def test_version_ranges_handle_local_and_prerelease_suffixes(self):
        self.assertTrue(
            preflight.version_in_range("2.14.0a0+gitabcdef0", (2, 5), None)
        )
        self.assertTrue(
            preflight.version_in_range("4.46.3", (4, 46), (4, 47))
        )
        self.assertFalse(
            preflight.version_in_range("4.47.0", (4, 46), (4, 47))
        )

    def test_optional_xformers_failures_are_nonblocking(self):
        checks = preflight._optional_checks(
            [
                preflight.Check(
                    "xformers.build_sm90",
                    preflight.FAIL,
                    "SM90 missing",
                )
            ]
        )

        self.assertEqual(checks[0].status, preflight.WARN)
        self.assertEqual(failures(checks), [])

    def test_triton_pin_files_include_version_and_commit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory)
            docker = source / ".ci" / "docker"
            commits = docker / "ci_commit_pins"
            commits.mkdir(parents=True)
            (docker / "triton_version.txt").write_text(
                "3.8.0\n",
                encoding="utf-8",
            )
            (commits / "triton.txt").write_text(
                "43422b04287ec4e774e2b1b9316b7eff44219b3f\n",
                encoding="utf-8",
            )

            pins = preflight.read_triton_pins(source)

        self.assertEqual(pins.version, "3.8.0")
        self.assertEqual(
            pins.commit,
            "43422b04287ec4e774e2b1b9316b7eff44219b3f",
        )
        self.assertTrue(
            preflight.commits_match("43422b04", pins.commit)
        )
        self.assertFalse(
            preflight.commits_match("deadbee", pins.commit)
        )


if __name__ == "__main__":
    unittest.main()
