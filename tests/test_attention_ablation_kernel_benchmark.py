"""CPU-only tests for the FlashAttention-paper kernel benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    NEOBERT_ROOT
    / "scripts"
    / "attention_ablation"
    / "benchmark_attention_kernels.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_attention_kernels_for_tests",
    SCRIPT,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {SCRIPT}")
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class TestPaperProtocols(unittest.TestCase):
    def test_default_grids_match_both_papers(self):
        args = benchmark.build_parser().parse_args([])
        benchmark._validate_args(args)
        cases = benchmark.generate_cases(args)
        fa1 = [case for case in cases if case.protocol == "fa1-e6"]
        fa2 = [case for case in cases if case.protocol == "fa2-4.1"]

        self.assertEqual(len(fa1), 11 * 10 * 2 * 2)
        self.assertEqual(len(fa2), 11 * 2 * 6 * 2)
        self.assertEqual(
            {case.variant for case in cases},
            set(benchmark.DEFAULT_VARIANTS),
        )
        self.assertEqual(
            sorted({case.sequence_length for case in fa1}),
            list(benchmark.FA1_SEQUENCE_LENGTHS),
        )
        self.assertEqual({case.batch_size for case in fa1}, {16})
        self.assertEqual({case.heads for case in fa1}, {8})
        self.assertEqual({case.head_dim for case in fa1}, {64})
        self.assertEqual({case.dropout_p for case in fa1}, {0.0, 0.1})
        self.assertEqual({case.padding_mask for case in fa1}, {False, True})
        self.assertEqual({case.causal for case in fa1}, {False})
        self.assertEqual({case.repetitions for case in fa1}, {100})

        self.assertEqual({case.total_tokens for case in fa2}, {16_384})
        self.assertEqual({case.hidden_size for case in fa2}, {2_048})
        self.assertEqual({case.head_dim for case in fa2}, {64, 128})
        self.assertEqual({case.causal for case in fa2}, {False, True})
        self.assertEqual({case.padding_mask for case in fa2}, {False})
        self.assertEqual({case.dropout_p for case in fa2}, {0.0})
        self.assertEqual({case.repetitions for case in fa2}, {30})

    def test_smoke_overrides_produce_one_requested_row(self):
        args = benchmark.build_parser().parse_args(
            [
                "--protocol",
                "fa2",
                "--variants",
                "real-torch",
                "--fa2-sequence-lengths",
                "512",
                "--fa2-head-dims",
                "64",
                "--fa2-causal-values",
                "false",
                "--fa2-repetitions",
                "2",
                "--warmup-repetitions",
                "1",
            ]
        )
        cases = benchmark.generate_cases(args)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.variant, "real-torch")
        self.assertEqual((case.batch_size, case.heads), (32, 32))
        self.assertEqual(case.repetitions, 2)

    def test_padding_mask_has_paper_valid_length_range_and_is_reproducible(self):
        case = benchmark.BenchmarkCase(
            protocol="fa1-e6",
            variant="real-torch",
            batch_size=16,
            heads=8,
            sequence_length=128,
            head_dim=64,
            causal=False,
            padding_mask=True,
            dropout_p=0.0,
            repetitions=1,
            warmup_repetitions=1,
        )
        first = benchmark.make_padding_mask(case, device=torch.device("cpu"), seed=17)
        second = benchmark.make_padding_mask(case, device=torch.device("cpu"), seed=17)
        torch.testing.assert_close(first, second)
        valid_lengths = first.logical_not().sum(dim=1)
        self.assertGreaterEqual(valid_lengths.min().item(), 108)
        self.assertLessEqual(valid_lengths.max().item(), 128)
        for row, valid_length in zip(first, valid_lengths):
            self.assertFalse(row[:valid_length].any())
            self.assertTrue(row[valid_length:].all())

    def test_flash_padding_mask_is_explicitly_unsupported(self):
        case = benchmark.BenchmarkCase(
            protocol="fa1-e6",
            variant="real-flash",
            batch_size=16,
            heads=8,
            sequence_length=128,
            head_dim=64,
            causal=False,
            padding_mask=True,
            dropout_p=0.0,
            repetitions=1,
            warmup_repetitions=1,
        )
        with self.assertRaisesRegex(benchmark.UnsupportedCase, "key-padding"):
            benchmark._attention_callable(case)

    def test_split_and_dual_flash_dropout_are_explicitly_unsupported(self):
        for variant in ("split-flash", "dual-flash"):
            with self.subTest(variant=variant):
                case = benchmark.BenchmarkCase(
                    protocol="fa1-e6",
                    variant=variant,
                    batch_size=16,
                    heads=8,
                    sequence_length=128,
                    head_dim=64,
                    causal=False,
                    padding_mask=False,
                    dropout_p=0.1,
                    repetitions=1,
                    warmup_repetitions=1,
                )
                with self.assertRaisesRegex(
                    benchmark.UnsupportedCase,
                    "shared dropout",
                ):
                    benchmark._attention_callable(case)


class TestAccounting(unittest.TestCase):
    def make_case(self, *, variant="real-torch", causal=False):
        return benchmark.BenchmarkCase(
            protocol="fa2-4.1",
            variant=variant,
            batch_size=32,
            heads=32,
            sequence_length=512,
            head_dim=64,
            causal=causal,
            padding_mask=False,
            dropout_p=0.0,
            repetitions=30,
            warmup_repetitions=10,
        )

    def test_nominal_flops_follow_paper_formula(self):
        case = self.make_case()
        expected_forward = 4 * 32 * 512**2 * 32 * 64
        flops = benchmark.paper_nominal_flops(case)
        self.assertEqual(flops["forward"], expected_forward)
        self.assertEqual(flops["backward"], 2.5 * expected_forward)
        self.assertEqual(flops["combined"], 3.5 * expected_forward)

        causal = benchmark.paper_nominal_flops(self.make_case(causal=True))
        self.assertEqual(causal["forward"], expected_forward / 2)

    def test_algebra_multipliers_and_physical_input_bytes_are_separate(self):
        real = benchmark._base_row(self.make_case(), element_size=2)
        complex_row = benchmark._base_row(
            self.make_case(variant="complex-torch"), element_size=2
        )
        dual = benchmark._base_row(
            self.make_case(variant="dual-torch"), element_size=2
        )
        self.assertEqual(real["logical_algebra_multiplier"], 1.0)
        self.assertEqual(complex_row["logical_algebra_multiplier"], 2.0)
        self.assertEqual(dual["logical_algebra_multiplier"], 3.0)
        self.assertEqual(complex_row["input_qkv_bytes"], 2 * real["input_qkv_bytes"])
        self.assertEqual(
            complex_row["paper_nominal_forward_flops"],
            real["paper_nominal_forward_flops"],
        )
        self.assertEqual(
            complex_row["logical_forward_flops"],
            2 * real["paper_nominal_forward_flops"],
        )
        self.assertEqual(real["backend_target"], "pytorch-sdpa-auto")
        self.assertEqual(
            complex_row["backend_target"],
            "pytorch-sdpa-auto-packed-complex",
        )
        split_flash = benchmark._base_row(
            self.make_case(variant="split-flash"), element_size=2
        )
        dual_flash = benchmark._base_row(
            self.make_case(variant="dual-flash"), element_size=2
        )
        self.assertIn("one-packed-split-complex", split_flash["backend_target"])
        self.assertEqual(
            dual_flash["backend_target"],
            "triton-fused-dual-flash",
        )
        self.assertIsNone(real["backend_effective"])

    def test_token_rates_use_batch_times_sequence_for_each_phase(self):
        total_tokens = self.make_case().total_tokens
        self.assertEqual(
            benchmark._tokens_per_second(total_tokens, 2.0),
            total_tokens / 0.002,
        )

    def test_failure_classification_distinguishes_oom_and_unsupported(self):
        self.assertEqual(
            benchmark._failure_status(RuntimeError("CUDA out of memory")), "oom"
        )
        self.assertEqual(
            benchmark._failure_status(benchmark.UnsupportedCase("missing op")),
            "unsupported",
        )
        for message in ("requires Triton", "requires compute capability 8+"):
            self.assertEqual(
                benchmark._failure_status(RuntimeError(message)), "unsupported"
            )
        self.assertEqual(benchmark._failure_status(RuntimeError("bad result")), "error")


class TestOutput(unittest.TestCase):
    def _resume_argv(self, output: Path) -> list[str]:
        return [
            "--protocol",
            "fa1",
            "--variants",
            "real-torch",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--fa1-sequence-lengths",
            "128,256",
            "--fa1-dropouts",
            "0",
            "--fa1-padding-mask-values",
            "false",
            "--fa1-repetitions",
            "1",
            "--warmup-repetitions",
            "1",
            "--output",
            str(output),
            "--resume",
            "--wandb-mode",
            "online",
            "--wandb-project",
            "resume-tests",
            "--wandb-entity",
            "test-entity",
            "--wandb-group",
            "test-group",
            "--wandb-name",
            "test-name",
            "--wandb-id",
            "test-id",
        ]

    @staticmethod
    def _finished_row(case):
        row = benchmark._base_row(case, element_size=2)
        row["status"] = "ok"
        return row

    def test_atomic_json_replaces_destination_without_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.json"
            benchmark.atomic_write_json(path, {"rows": [1], "status": "running"})
            benchmark.atomic_write_json(path, {"rows": [1, 2], "status": "complete"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"rows": [1, 2], "status": "complete"},
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_legacy_partial_resume_validates_exact_case_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            args = benchmark.build_parser().parse_args(self._resume_argv(output))
            cases = benchmark.generate_cases(args)
            legacy_cli = vars(args).copy()
            legacy_cli.pop("resume")
            legacy_cli["output"] = str(output.resolve())
            payload = {
                "schema_version": 1,
                "benchmark": "attention-kernels",
                "status": "running",
                "cli": legacy_cli,
                "device": {"requested_device": "cpu"},
                "rows": [self._finished_row(cases[0])],
            }

            loaded = benchmark._validate_resume_payload(
                payload,
                args=args,
                cases=cases,
                metadata={"requested_device": "cpu"},
            )
            self.assertEqual(len(loaded["rows"]), 1)

            incompatible = dict(payload)
            incompatible["rows"] = [dict(payload["rows"][0])]
            incompatible["rows"][0]["case_id"] = "different-case"
            with self.assertRaisesRegex(ValueError, "expected case prefix"):
                benchmark._validate_resume_payload(
                    incompatible,
                    args=args,
                    cases=cases,
                    metadata={"requested_device": "cpu"},
                )

            incompatible_args = benchmark.build_parser().parse_args(
                self._resume_argv(output) + ["--seed", "8"]
            )
            with self.assertRaisesRegex(ValueError, "incompatible with the requested"):
                benchmark._validate_resume_payload(
                    payload,
                    args=incompatible_args,
                    cases=benchmark.generate_cases(incompatible_args),
                    metadata={"requested_device": "cpu"},
                )

            device_payload = dict(payload)
            device_payload["device"] = {
                "requested_device": "cuda:0",
                "cuda_device_name": "NVIDIA A100-SXM4-80GB",
            }
            with self.assertRaisesRegex(ValueError, "cuda_device_name"):
                benchmark._validate_resume_payload(
                    device_payload,
                    args=args,
                    cases=cases,
                    metadata={
                        "requested_device": "cuda:0",
                        "cuda_device_name": "different GPU",
                    },
                )

    def test_resume_skips_prefix_and_logs_only_new_wandb_steps(self):
        class FakeRun:
            def __init__(self):
                self.logged_steps = []
                self.summary = {}
                self.name = "test-name"
                self.finished = False

            def log(self, payload, step=None):
                self.logged_steps.append(step)

            def finish(self):
                self.finished = True

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            argv = self._resume_argv(output)
            args = benchmark.build_parser().parse_args(argv)
            cases = benchmark.generate_cases(args)
            metadata = {"requested_device": "cpu"}
            payload = {
                "schema_version": 1,
                "benchmark": "attention-kernels",
                "status": "running",
                "started_at": "2026-01-01T00:00:00+00:00",
                "device": metadata,
                "cli": vars(args) | {"output": str(output.resolve())},
                "resume_signature": benchmark._resume_signature(args, cases),
                "rows": [self._finished_row(cases[0])],
            }
            benchmark.atomic_write_json(output, payload)
            run = FakeRun()

            def finish_case(case, *, device, seed):
                self.assertEqual(case, cases[1])
                self.assertEqual(seed, 8)
                return self._finished_row(case)

            with (
                mock.patch.object(benchmark, "device_metadata", return_value=metadata),
                mock.patch.object(
                    benchmark,
                    "benchmark_case",
                    side_effect=finish_case,
                ) as benchmark_case,
                mock.patch.object(benchmark, "_wandb_init", return_value=run),
                mock.patch.object(benchmark, "_wandb_log_results") as log_results,
            ):
                self.assertEqual(benchmark.main(argv), 0)

            benchmark_case.assert_called_once()
            self.assertEqual(run.logged_steps, [1])
            self.assertTrue(run.finished)
            log_results.assert_called_once()
            completed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(len(completed["rows"]), 2)
            self.assertEqual(completed["resume_history"][0]["completed_rows"], 1)

            with (
                mock.patch.object(benchmark, "device_metadata", return_value=metadata),
                mock.patch.object(benchmark, "benchmark_case") as benchmark_case,
                mock.patch.object(benchmark, "_wandb_init") as wandb_init,
            ):
                self.assertEqual(benchmark.main(argv), 0)
            benchmark_case.assert_not_called()
            wandb_init.assert_not_called()

    def test_cuda_allocator_note_does_not_claim_hbm_traffic(self):
        note = benchmark.MEMORY_MEASUREMENT_NOTE.lower()
        self.assertIn("not hbm traffic", note)
        self.assertIn("allocator", note)


if __name__ == "__main__":
    unittest.main()
