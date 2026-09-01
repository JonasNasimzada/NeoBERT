"""CPU-only regressions for OptiBERTneo startup and checkpoint handling."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from accelerate.utils import DistributedType
from omegaconf import OmegaConf
from transformers import AutoModelForMaskedLM

from neobert.model import NeoBERTConfig, NeoBERTLMHead
from neobert.optimizer import get_optimizer
from neobert.pretraining.metrics import Metrics
from neobert.pretraining.trainer import (
    RESUME_SIGNATURE_FILENAME,
    _state_dict_without_compile_prefix,
    build_prepacked_resume_signature,
    discover_resume_checkpoint,
    establish_prepacked_resume_signature,
    ensure_babylm_auto_map,
    resolve_resume_topology,
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

            def log(self, values, step=None):
                self.logged = values
                self.logged_step = step

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
        ensure_babylm_auto_map(model.config)

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
            self.assertEqual(
                loaded.config.auto_map,
                {
                    "AutoConfig": "model.NeoBERTConfig",
                    "AutoModelForMaskedLM": "model.NeoBERTLMHead",
                    "AutoModelForSequenceClassification": (
                        "model.NeoBERTHFForSequenceClassification"
                    ),
                },
            )
            auto_loaded = AutoModelForMaskedLM.from_pretrained(
                temporary_directory,
                trust_remote_code=True,
            )
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            labels = torch.tensor([[-100, 2, -100, 4]], dtype=torch.long)
            output = auto_loaded(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                token_type_ids=torch.zeros_like(input_ids),
                labels=labels,
            )
            self.assertEqual(
                tuple(output.logits.shape),
                (1, 4, config.vocab_size),
            )
            self.assertTrue(torch.isfinite(output.loss))


class TestPrepackedResumeSignature(unittest.TestCase):
    class DatasetStub:
        _fingerprint = "prepared-arrow-fingerprint"
        column_names = ["input_ids", "document_ids"]

        def __len__(self):
            return 1_269_760

    class BackendStub:
        def to_str(self):
            return '{"model":{"type":"BPE","vocab":{"a":0}}}'

    class TokenizerStub:
        backend_tokenizer = None
        model_max_length = 1_024
        bos_token_id = 0
        eos_token_id = 2
        pad_token_id = 1
        mask_token_id = 50_264
        unk_token_id = 3
        sep_token_id = 2
        cls_token_id = 0

        def __init__(self):
            self.backend_tokenizer = TestPrepackedResumeSignature.BackendStub()

        def __len__(self):
            return 50_265

    def make_fixture(self, root):
        dataset_path = root / "dataset"
        dataset_path.mkdir()
        manifest = {
            "format_version": 1,
            "dataset_fingerprint": self.DatasetStub._fingerprint,
            "rows": len(self.DatasetStub()),
            "sequence_length": 1_024,
            "tokenizer": {
                "name": "FacebookAI/roberta-base",
                "revision": "tokenizer-commit",
                "vocab_size": 50_265,
                "bos_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 1,
                "mask_token_id": 50_264,
            },
        }
        (dataset_path / "optibertneo_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cfg = OmegaConf.create(
            {
                "seed": 0,
                "dataset": {"path_to_disk": str(dataset_path)},
                "dataloader": {"train": {"batch_size": 8}},
                "datacollator": {
                    "mlm_probability": 0.20,
                    "mask_all": True,
                },
                "trainer": {
                    "gradient_accumulation_steps": 32,
                    "max_steps": 620,
                    "mixed_precision": "bf16",
                    "tf32": True,
                    "gradient_clipping": 1,
                    "compile": True,
                    "compile_fullgraph": False,
                    "find_unused_parameters": False,
                },
                "model": {
                    "hidden_size": 768,
                    "num_hidden_layers": 28,
                    "attention_space": "multispace",
                    "attention_backend": "flex",
                },
                "tokenizer": {
                    # A staged local path is deliberately absent from the
                    # signature; the pinned manifest source is its identity.
                    "pretrained_model_name_or_path": str(dataset_path / "tokenizer"),
                    "revision": None,
                    "max_length": 1_024,
                    "vocab_size": 50_265,
                    "truncation": True,
                    "chunk_long_documents": True,
                    "trust_remote_code": False,
                },
                "optimizer": {
                    "name": "AdamW",
                    "hparams": {"lr": 6e-4, "betas": [0.9, 0.95]},
                },
                "scheduler": {
                    "warmup_steps": 500,
                    "decay_steps": 620,
                    "decay": "cosine",
                    "final_ratio": 0.1,
                },
            }
        )
        signature = build_prepacked_resume_signature(
            cfg,
            self.DatasetStub(),
            self.TokenizerStub(),
            world_size=8,
            topology={
                "world_size": 8,
                "num_machines": 2,
                "processes_per_machine": 4,
            },
        )
        return cfg, signature

    def test_signature_contains_every_resume_critical_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, signature = self.make_fixture(Path(temporary_directory))

        self.assertEqual(
            signature["topology"],
            {
                "world_size": 8,
                "num_machines": 2,
                "processes_per_machine": 4,
            },
        )
        self.assertEqual(
            signature["batching"],
            {
                "per_device_microbatch": 8,
                "gradient_accumulation_steps": 32,
            },
        )
        self.assertEqual(
            signature["dataset"]["fingerprint"],
            self.DatasetStub._fingerprint,
        )
        self.assertEqual(len(signature["dataset"]["manifest_sha256"]), 64)
        self.assertEqual(signature["seed"], 0)
        for identity in (
            "model",
            "data_pipeline",
            "tokenizer",
            "optimizer",
            "scheduler",
            "trainer_semantics",
        ):
            self.assertIn(identity, signature)
        self.assertEqual(
            signature["tokenizer"]["source"],
            {
                "name": "FacebookAI/roberta-base",
                "revision": "tokenizer-commit",
            },
        )
        self.assertEqual(
            len(signature["tokenizer"]["runtime"]["backend_sha256"]),
            64,
        )

    def test_fresh_signature_is_atomically_established_and_reused(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, signature = self.make_fixture(root)
            run_root = root / "run"

            path = establish_prepacked_resume_signature(
                run_root,
                signature,
                checkpoint_exists=False,
            )
            second_path = establish_prepacked_resume_signature(
                run_root,
                copy.deepcopy(signature),
                checkpoint_exists=True,
            )

            self.assertEqual(path, run_root / RESUME_SIGNATURE_FILENAME)
            self.assertEqual(second_path, path)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                signature,
            )
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in run_root.iterdir())
            )

    def test_checkpoint_rejects_a_missing_signature(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, signature = self.make_fixture(root)
            run_root = root / "run"

            with self.assertRaisesRegex(
                RuntimeError,
                "checkpoint state exists but resume signature is missing",
            ):
                establish_prepacked_resume_signature(
                    run_root,
                    signature,
                    checkpoint_exists=True,
                )
            self.assertFalse((run_root / RESUME_SIGNATURE_FILENAME).exists())

    def test_each_resume_critical_mismatch_is_rejected(self):
        mutations = {
            "topology": lambda value: value["topology"].update(world_size=4),
            "microbatch": lambda value: value["batching"].update(
                per_device_microbatch=4
            ),
            "accumulation": lambda value: value["batching"].update(
                gradient_accumulation_steps=64
            ),
            "dataset": lambda value: value["dataset"].update(
                fingerprint="different-dataset"
            ),
            "manifest": lambda value: value["dataset"].update(
                manifest_sha256="0" * 64
            ),
            "seed": lambda value: value.update(seed=1),
            "model": lambda value: value["model"].update(hidden_size=512),
            "tokenizer": lambda value: value["tokenizer"]["runtime"].update(
                backend_sha256="1" * 64
            ),
            "optimizer": lambda value: value["optimizer"].update(name="Adam"),
            "scheduler": lambda value: value["scheduler"].update(
                warmup_steps=100
            ),
            "masking": lambda value: value["data_pipeline"][
                "datacollator"
            ].update(mlm_probability=0.15),
            "trainer": lambda value: value["trainer_semantics"].update(
                gradient_clipping=0.5
            ),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, signature = self.make_fixture(root)
            run_root = root / "run"
            establish_prepacked_resume_signature(
                run_root,
                signature,
                checkpoint_exists=False,
            )

            for name, mutate in mutations.items():
                with self.subTest(identity=name):
                    changed = copy.deepcopy(signature)
                    mutate(changed)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "resume signature mismatch",
                    ):
                        establish_prepacked_resume_signature(
                            run_root,
                            changed,
                            checkpoint_exists=True,
                        )

    def test_topology_resolution_is_rank_invariant_and_strict(self):
        environment = {"NUM_MACHINES": "2", "LOCAL_WORLD_SIZE": "4"}
        self.assertEqual(
            resolve_resume_topology(8, environment),
            {
                "world_size": 8,
                "num_machines": 2,
                "processes_per_machine": 4,
            },
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            resolve_resume_topology(4, environment)


if __name__ == "__main__":
    unittest.main()
