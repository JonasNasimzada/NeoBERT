"""Regression tests for the paper-style padding-free data packer."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_PATH = (
    NEOBERT_ROOT
    / "scripts"
    / "pretraining"
    / "preprocess.py"
)
SPEC = importlib.util.spec_from_file_location(
    "optibertneo_preprocess_for_tests",
    PREPROCESS_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {PREPROCESS_PATH}")
preprocess = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preprocess
SPEC.loader.exec_module(preprocess)


class TestPaddingFreePacking(unittest.TestCase):
    def test_documents_are_concatenated_without_padding(self):
        source = Dataset.from_dict(
            {
                "input_ids": [
                    [0, 10, 2],
                    [0, 20, 21, 2],
                    [0, 30, 31, 32, 2],
                ]
            }
        )

        with tempfile.TemporaryDirectory() as cache_directory:
            packed = preprocess.pack_tokenized_dataset(
                source,
                sequence_length=4,
                cache_dir=cache_directory,
            )
            self.assertEqual(len(packed), 3)
            self.assertEqual(
                packed["input_ids"],
                [
                    [0, 10, 2, 0],
                    [20, 21, 2, 0],
                    [30, 31, 32, 2],
                ],
            )
            self.assertEqual(
                packed["document_ids"],
                [
                    [0, 0, 0, 1],
                    [1, 1, 1, 2],
                    [2, 2, 2, 2],
                ],
            )
            self.assertTrue(
                all(
                    document_id >= 0
                    for row in packed["document_ids"]
                    for document_id in row
                )
            )

    def test_only_the_incomplete_tail_is_dropped(self):
        source = Dataset.from_dict(
            {
                "input_ids": [
                    [0, 11, 2],
                    [0, 12],
                ]
            }
        )

        with tempfile.TemporaryDirectory() as cache_directory:
            packed = preprocess.pack_tokenized_dataset(
                source,
                sequence_length=4,
                cache_dir=cache_directory,
            )
            self.assertEqual(len(packed), 1)
            self.assertEqual(packed[0]["input_ids"], [0, 11, 2, 0])
            self.assertEqual(packed[0]["document_ids"], [0, 0, 0, 1])

    def test_cross_document_packing_has_only_full_input_id_rows(self):
        source = Dataset.from_dict(
            {
                "input_ids": [
                    [101, 10, 102],
                    [101, 20, 21, 102],
                    [101, 30, 31, 32, 102],
                ]
            }
        )

        with tempfile.TemporaryDirectory() as cache_directory:
            packed = preprocess.pack_tokenized_dataset(
                source,
                sequence_length=4,
                cache_dir=cache_directory,
                cross_document_attention=True,
            )

            self.assertEqual(packed.column_names, ["input_ids"])
            self.assertNotIn("document_ids", packed.column_names)
            self.assertEqual(
                packed["input_ids"],
                [
                    [101, 10, 102, 101],
                    [20, 21, 102, 101],
                    [30, 31, 32, 102],
                ],
            )
            self.assertTrue(
                all(len(row) == 4 for row in packed["input_ids"])
            )

    def test_source_row_validation_split_is_deterministic(self):
        source = Dataset.from_dict(
            {
                "row_id": list(range(100)),
                "text": [f"row {index}" for index in range(100)],
            }
        )

        first = preprocess.create_train_validation_split(
            source,
            validation_fraction=0.2,
            seed=17,
        )
        second = preprocess.create_train_validation_split(
            source,
            validation_fraction=0.2,
            seed=17,
        )

        self.assertIsInstance(first, DatasetDict)
        self.assertEqual(first["train"]["row_id"], second["train"]["row_id"])
        self.assertEqual(
            first["validation"]["row_id"],
            second["validation"]["row_id"],
        )
        self.assertEqual(len(first["train"]), 80)
        self.assertEqual(len(first["validation"]), 20)
        self.assertFalse(
            set(first["train"]["row_id"])
            & set(first["validation"]["row_id"])
        )

    def test_atomic_save_records_manifest_and_local_tokenizer(self):
        class TokenizerStub:
            bos_token_id = 0
            pad_token_id = 1
            eos_token_id = 2
            mask_token_id = 50_264

            def __len__(self):
                return 50_265

            def save_pretrained(self, path):
                path.mkdir(parents=True)
                (path / "tokenizer.json").write_text(
                    "{}\n",
                    encoding="utf-8",
                )

        dataset = Dataset.from_dict(
            {
                "input_ids": [[0, 10, 11, 2], [0, 20, 21, 2]],
                "document_ids": [[0, 0, 0, 0], [1, 1, 1, 1]],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "prepared"
            config = OmegaConf.create(
                {
                    "dataset": {
                        "name": "fineweb_edu",
                        "path_to_disk": str(output_path),
                        "pack_to_length": 4,
                        "approx_token_limit": 8,
                        "train": {
                            "path": "HuggingFaceFW/fineweb-edu",
                            "name": "sample-10BT",
                            "split": "train",
                            "revision": "v1.0.0",
                        },
                    },
                    "tokenizer": {
                        "pretrained_model_name_or_path": (
                            "FacebookAI/roberta-base"
                        ),
                        "revision": "test-revision",
                    },
                }
            )

            preprocess.save_preprocessed_dataset(
                dataset,
                TokenizerStub(),
                config,
            )

            self.assertTrue((output_path / "dataset_info.json").is_file())
            self.assertTrue(
                (output_path / "tokenizer" / "tokenizer.json").is_file()
            )
            manifest = json.loads(
                (output_path / "optibertneo_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["sequence_length"], 4)
            self.assertEqual(manifest["packed_token_positions"], 8)
            self.assertTrue(manifest["packing"]["padding_free"])
            self.assertFalse(
                manifest["packing"]["cross_document_attention"]
            )
            self.assertEqual(
                manifest["source"]["revision"],
                "v1.0.0",
            )
            self.assertEqual(
                manifest["tokenizer"]["revision"],
                "test-revision",
            )
            with self.assertRaises(FileExistsError):
                preprocess.save_preprocessed_dataset(
                    dataset,
                    TokenizerStub(),
                    config,
                )

    def test_dataset_dict_manifest_records_flat_split_counts(self):
        class TokenizerStub:
            bos_token_id = None
            pad_token_id = 0
            eos_token_id = None
            mask_token_id = 103

            def __len__(self):
                return 30_522

            def save_pretrained(self, path):
                path.mkdir(parents=True)
                (path / "tokenizer.json").write_text(
                    "{}\n",
                    encoding="utf-8",
                )

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"input_ids": [[101, 10, 11, 102], [101, 20, 21, 102]]}
                ),
                "validation": Dataset.from_dict(
                    {"input_ids": [[101, 30, 31, 102]]}
                ),
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "prepared"
            config = OmegaConf.create(
                {
                    "dataset": {
                        "name": "babylm_2026_strict",
                        "path_to_disk": str(output_path),
                        "pack_to_length": 4,
                        "cross_document_attention": True,
                        "train_split": "train",
                        "train": {
                            "path": "BabyLM-community/BabyLM-2026-Strict",
                            "split": "train",
                            "revision": "test-revision",
                        },
                    },
                    "tokenizer": {
                        "pretrained_model_name_or_path": (
                            "google-bert/bert-base-uncased"
                        ),
                        "revision": None,
                    },
                }
            )

            preprocess.save_preprocessed_dataset(
                dataset,
                TokenizerStub(),
                config,
            )

            self.assertTrue((output_path / "dataset_dict.json").is_file())
            manifest = json.loads(
                (output_path / "optibertneo_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["packed_token_positions"], 8)
            self.assertEqual(manifest["splits"]["train"]["rows"], 2)
            self.assertEqual(manifest["splits"]["train"]["tokens"], 8)
            self.assertEqual(
                manifest["splits"]["validation"]["rows"],
                1,
            )
            self.assertEqual(
                manifest["splits"]["validation"]["tokens"],
                4,
            )
            self.assertEqual(
                manifest["splits"]["train"]["columns"],
                ["input_ids"],
            )
            self.assertTrue(manifest["packing"]["padding_free"])
            self.assertTrue(
                manifest["packing"]["cross_document_attention"]
            )
            self.assertFalse(manifest["packing"]["document_ids"])


if __name__ == "__main__":
    unittest.main()
