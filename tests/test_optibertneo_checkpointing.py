"""CPU-only regressions for OptiBERTneo startup and checkpoint handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from accelerate.utils import DistributedType

from neobert.model import NeoBERTConfig, NeoBERTLMHead
from neobert.optimizer import get_optimizer
from neobert.pretraining.metrics import Metrics
from neobert.pretraining.trainer import (
    _state_dict_without_compile_prefix,
    discover_resume_checkpoint,
)


class TestOptiBERTneoCheckpointing(unittest.TestCase):
    def test_metric_counters_remain_exact_python_integers(self):
        class AcceleratorStub:
            device = torch.device("cpu")

            def reduce(self, value, reduction):
                if reduction != "sum":  # pragma: no cover
                    raise AssertionError(f"unexpected reduction: {reduction}")
                if value.dtype != torch.float64:  # pragma: no cover
                    raise AssertionError(f"unexpected dtype: {value.dtype}")
                return value

            def log(self, values):
                self.logged = values

        accelerator = AcceleratorStub()
        metrics = Metrics()
        metrics["train/local_samples"] = 16_777_217
        metrics["train/local_tokens"] = 1_300_234_240
        metrics["train/local_num_pred"] = 260_046_849
        metrics["train/local_sum_loss"] = 2_600_468_490.0
        metrics["train/local_num_correct"] = 1

        metrics.log(accelerator)

        for name, expected in (
            ("train/samples", 16_777_217),
            ("train/tokens", 1_300_234_240),
            ("train/masked_tokens", 260_046_849),
        ):
            self.assertEqual(metrics[name], expected)
            self.assertIsInstance(metrics[name], int)

    def test_adamw_does_not_import_the_optional_soap_optimizer(self):
        model = torch.nn.Linear(4, 4)

        optimizer = get_optimizer(
            model,
            DistributedType.NO,
            name="AdamW",
            lr=6e-4,
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_resume_uses_latest_complete_checkpoint_and_skips_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "checkpoint_0").mkdir()
            complete = root / "checkpoint_1"
            complete.mkdir()
            (complete / "_SUCCESS").write_text(
                "complete\n",
                encoding="utf-8",
            )
            (root / "checkpoint_2").mkdir()
            (root / "unrelated").mkdir()

            resume_path, next_iteration = discover_resume_checkpoint(root)

            self.assertEqual(resume_path, complete)
            self.assertEqual(next_iteration, 3)

    def test_compile_prefix_is_removed_recursively(self):
        value = torch.tensor([1.0])
        stripped = _state_dict_without_compile_prefix(
            {"_orig_mod._orig_mod.weight": value}
        )

        self.assertEqual(tuple(stripped), ("weight",))
        self.assertIs(stripped["weight"], value)

    def test_tied_lm_head_exports_with_safe_serialization(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            hidden_act="swiglu",
            fused_swiglu=False,
            vocab_size=32,
            max_length=8,
            rope=False,
            tie_word_embeddings=True,
            lm_head_bias=False,
            attention_space="real",
            attention_backend="torch",
        )
        model = NeoBERTLMHead(config)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model.save_pretrained(
                temporary_directory,
                safe_serialization=True,
            )
            self.assertTrue(
                (Path(temporary_directory) / "model.safetensors").is_file()
            )
            loaded = NeoBERTLMHead.from_pretrained(temporary_directory)
            self.assertIs(
                loaded.decoder.weight,
                loaded.model.encoder.weight,
            )


if __name__ == "__main__":
    unittest.main()
