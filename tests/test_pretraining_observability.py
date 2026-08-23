"""CPU-only tests for pretraining validation and experiment telemetry."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from neobert.pretraining.metrics import Metrics
from neobert.pretraining.trainer import (
    advance_gradient_accumulation,
    build_training_summary,
    evaluate_mlm,
    max_time_reached,
    split_train_validation_dataset,
    validate_flat_packed_dataset_dict,
    validation_dataloader_kwargs,
    wandb_init_kwargs,
)
from neobert.dataloader.dataloader import get_dataloader


class AcceleratorStub:
    device = torch.device("cpu")
    process_index = 0
    num_processes = 1
    mixed_precision = "no"
    distributed_type = "NO"

    def __init__(self):
        self.logged = None
        self.logged_step = None
        self.reductions = []

    def reduce(self, value, reduction):
        self.reductions.append(reduction)
        return value

    def log(self, values, step=None):
        self.logged = values
        self.logged_step = step


class StochasticValidationLoader:
    """Mimic worker-free MLM collation driven by the global CPU RNG."""

    def __iter__(self):
        original = torch.tensor(
            [[1, 2, 3, 4], [4, 3, 2, 1]],
            dtype=torch.long,
        )
        for _ in range(2):
            selected = torch.rand(original.shape) < 0.5
            yield {
                "input_ids": torch.where(selected, 7, original),
                "labels": torch.where(
                    selected,
                    original,
                    torch.full_like(original, -100),
                ),
            }


class FixedLogitModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, document_ids=None):
        del attention_mask, document_ids
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, 8)
        predictions = (
            torch.arange(sequence_length).view(1, -1).expand(batch_size, -1)
            + 1
        ) % logits.shape[-1]
        logits.scatter_(-1, predictions.unsqueeze(-1), 3.0)
        return {"logits": logits}


class ValueCollator:
    def __call__(self, features):
        return {"values": torch.tensor([feature["value"] for feature in features])}


class TestMetricsLogging(unittest.TestCase):
    def test_log_is_explicit_flat_and_hides_rank_local_counters(self):
        accelerator = AcceleratorStub()
        metrics = Metrics()
        metrics["train/steps"] = 7
        metrics["train/local_samples"] = 2
        metrics["train/local_tokens"] = 8
        metrics["train/local_num_pred"] = 4
        metrics["train/local_sum_loss"] = 8.0
        metrics["train/local_num_correct"] = 1

        payload = metrics.log(
            accelerator,
            step=7,
            extra_metrics={"validation/mlm_loss": 1.5},
        )

        self.assertEqual(accelerator.logged_step, 7)
        self.assertEqual(payload["train/tokens"], 8)
        self.assertEqual(payload["train/masked_tokens"], 4)
        self.assertEqual(payload["train/loss"], 2.0)
        self.assertEqual(payload["validation/mlm_loss"], 1.5)
        self.assertFalse(any("/local_" in key for key in payload))
        self.assertFalse(any("/local_" in key for key in metrics))
        self.assertEqual(accelerator.reductions, ["sum"])


class TestDatasetAndValidation(unittest.TestCase):
    def test_datasetdict_uses_named_splits_and_dataset_remains_legacy_train(self):
        train = Dataset.from_dict({"input_ids": [[1, 2]]})
        validation = Dataset.from_dict({"input_ids": [[3, 4]]})

        selected_train, selected_validation = split_train_validation_dataset(
            DatasetDict({"train": train, "validation": validation})
        )
        self.assertIs(selected_train, train)
        self.assertIs(selected_validation, validation)

        selected_train, selected_validation = split_train_validation_dataset(train)
        self.assertIs(selected_train, train)
        self.assertIsNone(selected_validation)

        with self.assertRaisesRegex(ValueError, "train"):
            split_train_validation_dataset(DatasetDict({"validation": validation}))

    def test_validation_loader_forces_worker_free_stable_order(self):
        cfg = OmegaConf.create(
            {
                "dataloader": {
                    "train": {
                        "batch_size": 8,
                        "shuffle": True,
                        "num_workers": 6,
                        "persistent_workers": True,
                    },
                    "validation": {"batch_size": 4, "num_workers": 9},
                }
            }
        )

        options = validation_dataloader_kwargs(cfg)

        self.assertEqual(options["batch_size"], 4)
        self.assertFalse(options["shuffle"])
        self.assertEqual(options["num_workers"], 0)
        self.assertFalse(options["persistent_workers"])

    def test_flat_packed_ablation_dataset_and_manifest_are_cross_checked(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]]}
                ),
                "validation": Dataset.from_dict(
                    {"input_ids": [[9, 10, 11, 12]]}
                ),
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = {
                "sequence_length": 4,
                "source_rows": 3,
                "packing": {
                    "padding_free": True,
                    "cross_document_attention": True,
                    "document_ids": False,
                },
                "splits": {
                    "train": {
                        "rows": 2,
                        "tokens": 8,
                        "packed_token_positions": 8,
                        "columns": ["input_ids"],
                    },
                    "validation": {
                        "rows": 1,
                        "tokens": 4,
                        "packed_token_positions": 4,
                        "columns": ["input_ids"],
                    },
                },
            }
            path = Path(temporary_directory)
            (path / "optibertneo_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            cfg = OmegaConf.create(
                {
                    "dataset": {
                        "path_to_disk": str(path),
                        "pack_to_length": 4,
                        "cross_document_attention": True,
                        "expected_source_rows": 3,
                    }
                }
            )

            validate_flat_packed_dataset_dict(dataset, cfg)

            manifest["source_rows"] = 2
            (path / "optibertneo_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source row count"):
                validate_flat_packed_dataset_dict(dataset, cfg)

            manifest["source_rows"] = 3
            manifest["splits"]["validation"]["rows"] = 2
            (path / "optibertneo_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "validation row count"):
                validate_flat_packed_dataset_dict(dataset, cfg)

    def test_flat_packed_ablation_rejects_document_ids_and_short_rows(self):
        cfg = OmegaConf.create(
            {
                "dataset": {
                    "path_to_disk": "/unused",
                    "pack_to_length": 4,
                    "cross_document_attention": True,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "only input_ids"):
            validate_flat_packed_dataset_dict(
                DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "input_ids": [[1, 2, 3, 4]],
                                "document_ids": [[0, 0, 0, 0]],
                            }
                        ),
                        "validation": Dataset.from_dict(
                            {"input_ids": [[1, 2, 3, 4]]}
                        ),
                    }
                ),
                cfg,
            )

        with self.assertRaisesRegex(ValueError, "has 3 tokens"):
            validate_flat_packed_dataset_dict(
                DatasetDict(
                    {
                        "train": Dataset.from_dict({"input_ids": [[1, 2, 3]]}),
                        "validation": Dataset.from_dict(
                            {"input_ids": [[1, 2, 3, 4]]}
                        ),
                    }
                ),
                cfg,
            )

    def test_validation_is_repeatable_and_restores_rng_and_training_mode(self):
        accelerator = AcceleratorStub()
        model = FixedLogitModel()
        model.train()
        torch.manual_seed(12345)
        rng_before = torch.random.get_rng_state().clone()

        first = evaluate_mlm(
            accelerator,
            model,
            StochasticValidationLoader(),
            seed=99,
        )
        rng_after_first = torch.random.get_rng_state().clone()
        second = evaluate_mlm(
            accelerator,
            model,
            StochasticValidationLoader(),
            seed=99,
        )
        rng_after_second = torch.random.get_rng_state().clone()

        comparable_keys = set(first).difference({"validation/wall_time_seconds"})
        self.assertEqual(
            {key: first[key] for key in comparable_keys},
            {key: second[key] for key in comparable_keys},
        )
        self.assertTrue(torch.equal(rng_before, rng_after_first))
        self.assertTrue(torch.equal(rng_before, rng_after_second))
        self.assertTrue(model.training)
        self.assertGreater(first["validation/masked_tokens"], 0)


class TestRunMetadataAndTimeLimit(unittest.TestCase):
    def test_short_epoch_tail_does_not_advance_gradient_accumulation(self):
        metrics = Metrics()
        target_batch_size = 8
        update_groups = []
        processed_since_update = 0

        for epoch_batch_sizes in (
            (8, 8, 8, 3),
            (8, 8, 8, 8, 8),
        ):
            for batch_size in epoch_batch_sizes:
                # This counter remains the raw dataloader position used by
                # checkpoint resume, including the skipped short tail.
                metrics["train/batches"] += 1
                if batch_size < target_batch_size:
                    continue

                processed_since_update += 1
                if advance_gradient_accumulation(metrics, 4):
                    update_groups.append(processed_since_update)
                    processed_since_update = 0

        self.assertEqual(update_groups, [4, 4])
        self.assertEqual(processed_since_update, 0)
        self.assertEqual(metrics["train/batches"], 9)
        self.assertEqual(metrics["train/processed_batches"], 8)

    def test_accumulation_counter_resumes_without_changing_data_position(self):
        checkpointed = Metrics()
        checkpointed["train/batches"] = 9
        checkpointed["train/steps"] = 2
        checkpointed["train/processed_batches"] = 8

        resumed = Metrics()
        resumed.load_state_dict(checkpointed.state_dict())

        self.assertFalse(advance_gradient_accumulation(resumed, 4))
        self.assertFalse(advance_gradient_accumulation(resumed, 4))
        self.assertFalse(advance_gradient_accumulation(resumed, 4))
        self.assertTrue(advance_gradient_accumulation(resumed, 4))
        self.assertEqual(resumed["train/batches"], 9)
        self.assertEqual(resumed["train/processed_batches"], 12)

    def test_legacy_checkpoint_derives_accumulation_phase_from_steps(self):
        metrics = Metrics()
        metrics["train/batches"] = 9
        metrics["train/steps"] = 2

        self.assertFalse(advance_gradient_accumulation(metrics, 4))
        self.assertEqual(metrics["train/processed_batches"], 9)
        self.assertEqual(metrics["train/batches"], 9)

    def test_wandb_identity_is_stable_resolved_and_credential_free(self):
        cfg = OmegaConf.create(
            {
                "seed": 3,
                "experiment_name": "${wandb.project}",
                "wandb": {
                    "name": "complex/native",
                    "project": "complex-attention",
                    "entity": None,
                    "tags": ["complex", "native"],
                    "dir": "/tmp/wandb",
                    "mode": "offline",
                    "resume": "allow",
                    "group": "backend-comparison",
                    "job_type": "pretrain",
                },
            }
        )
        environment = {
            "SLURM_ARRAY_JOB_ID": "1234",
            "SLURM_ARRAY_TASK_ID": "2",
            "SLURM_JOB_PARTITION": "slowlane",
            "WANDB_API_KEY": "must-not-be-logged",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            options = wandb_init_kwargs(cfg, AcceleratorStub())

        self.assertEqual(options["id"], "1234-2-complex-native")
        self.assertEqual(options["group"], "backend-comparison")
        self.assertEqual(options["job_type"], "pretrain")
        self.assertEqual(options["config"]["experiment_name"], "complex-attention")
        self.assertEqual(
            options["config"]["runtime"]["slurm"]["job_partition"],
            "slowlane",
        )
        self.assertNotIn(
            "WANDB_API_KEY",
            str(options["config"]),
        )

    def test_time_limit_uses_an_any_rank_sum_reduction(self):
        accelerator = AcceleratorStub()
        with mock.patch(
            "neobert.pretraining.trainer.time.monotonic",
            return_value=11.0,
        ):
            reached = max_time_reached(
                accelerator,
                run_started_at=1.0,
                max_time_seconds=10.0,
            )

        self.assertTrue(reached)
        self.assertEqual(accelerator.reductions, ["sum"])

    def test_training_summary_distinguishes_complete_and_guard_limited_runs(self):
        cfg = OmegaConf.create({"trainer": {"max_steps": 10}})
        metrics = Metrics()
        metrics["train/steps"] = 8
        metrics["train/stopped_for_max_time"] = 1

        partial = build_training_summary(cfg, metrics)
        self.assertFalse(partial["train/completed_schedule"])
        self.assertTrue(partial["train/stopped_for_max_time"])

        metrics["train/steps"] = 10
        metrics["train/stopped_for_max_time"] = 0
        complete = build_training_summary(cfg, metrics)
        self.assertTrue(complete["train/completed_schedule"])
        self.assertFalse(complete["train/stopped_for_max_time"])

    def test_seeded_loader_is_independent_of_parent_rng_consumption(self):
        dataset = Dataset.from_dict({"value": list(range(12))})

        def collect_after_parent_draws(parent_draws):
            with mock.patch(
                "neobert.dataloader.dataloader.get_collator",
                return_value=ValueCollator(),
            ):
                dataloader = get_dataloader(
                    dataset,
                    tokenizer=object(),
                    batch_size=4,
                    shuffle=True,
                    num_workers=0,
                    persistent_workers=False,
                    seed=2718,
                )
                self.assertIsNotNone(dataloader.generator)
                self.assertEqual(dataloader.generator.initial_seed(), 2718)
                torch.manual_seed(0)
                torch.rand(parent_draws)
                return [batch["values"].tolist() for batch in dataloader]

        self.assertEqual(
            collect_after_parent_draws(1),
            collect_after_parent_draws(10_000),
        )


if __name__ == "__main__":
    unittest.main()
