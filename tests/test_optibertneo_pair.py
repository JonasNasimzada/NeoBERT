"""Static contracts for the parameter-matched OptiBERTneo model pair."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    NEOBERT_ROOT
    / "scripts"
    / "pretraining"
    / "validate_optibertneo_pair.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_optibertneo_pair_for_tests",
    VALIDATOR_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


UNSAFE_FULL_SUBMISSION_ENVIRONMENT = (
    "SMOKE_TEST",
    "GLOBAL_SEQUENCES",
    "MICRO_BATCH",
    "NUM_MACHINES",
    "GPUS_PER_NODE",
    "MACHINE_RANK",
    "EXPECTED_NUM_MACHINES",
    "EXPECTED_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "ACCELERATE_CONFIG",
)


def load_config(group: str, name: str) -> dict:
    path = NEOBERT_ROOT / "conf" / group / f"{name}.yaml"
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict):  # pragma: no cover
        raise TypeError(f"{path} must contain a mapping")
    return values


class TestOptiBERTneoModelPair(unittest.TestCase):
    def test_real_and_multispace_configs_share_the_paper_geometry(self):
        real = load_config("model", "optibertneo-198m")
        multispace = load_config("model", "optibertneo-198m-multispace")
        shared = {
            "hidden_size": 768,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "rope": True,
            "rms_norm": True,
            "embedding_rms_norm": True,
            "hidden_act": "swiglu",
            "fused_swiglu": False,
            "dropout": 0,
            "attention_dropout": 0,
            "tie_word_embeddings": True,
            "lm_head_bias": False,
            "ngpt": False,
            "attention_backend": "flex",
        }
        for key, expected in shared.items():
            with self.subTest(key=key):
                self.assertEqual(real[key], expected)
                self.assertEqual(multispace[key], expected)

        self.assertEqual(real["attention_space"], "real")
        self.assertEqual(real["intermediate_size"], 3_072)
        self.assertEqual(multispace["attention_space"], "multispace")
        self.assertEqual(multispace["intermediate_size"], 1_536)
        self.assertTrue(multispace["multispace_cuda_streams"])

        for contract in validator.CONTRACTS:
            config = validator._load_config(contract)
            self.assertEqual(
                config.attention_spaces,
                [contract.attention_space] * 28,
            )
            self.assertEqual(config.attention_backends, ["flex"] * 28)
            self.assertEqual(config.dim_head, 64)
        self.assertEqual(12 // 3, validator.HEADS_PER_SPACE)

    def test_exact_parameter_match_uses_the_papers_nonembedding_convention(self):
        hidden_size = 768
        layers = 28
        vocabulary_size = 50_265

        real_attention = 4 * hidden_size**2
        real_ffn = 3 * hidden_size * 2_048
        multispace_attention = 8 * hidden_size**2
        multispace_ffn = 3 * hidden_size * 1_024
        normalization = 2 * hidden_size
        real_block = real_attention + real_ffn + normalization
        multispace_block = (
            multispace_attention + multispace_ffn + normalization
        )
        paper_matrix_count = 12 * hidden_size**2 * layers
        non_embedding = layers * real_block + 2 * hidden_size
        embedding = vocabulary_size * hidden_size
        total = embedding + non_embedding

        self.assertEqual(real_attention, 2_359_296)
        self.assertEqual(multispace_attention, 4_718_592)
        self.assertEqual(real_ffn, 4_718_592)
        self.assertEqual(multispace_ffn, 2_359_296)
        self.assertEqual(real_block, multispace_block)
        self.assertEqual(real_block, validator.EXPECTED_BLOCK_PARAMETERS)
        self.assertEqual(paper_matrix_count, 198_180_864)
        self.assertEqual(
            non_embedding,
            validator.EXPECTED_NON_EMBEDDING_PARAMETERS,
        )
        self.assertEqual(embedding, validator.EXPECTED_EMBEDDING_PARAMETERS)
        self.assertEqual(total, validator.EXPECTED_TOTAL_PARAMETERS)

    def test_validator_constructs_both_full_graphs_on_meta(self):
        model_class = validator.NeoBERTLMHead
        allocation_devices = []

        def record_allocation(config):
            model = model_class(config)
            allocation_devices.append(next(model.parameters()).device.type)
            return model

        with mock.patch.object(
            validator,
            "NeoBERTLMHead",
            side_effect=record_allocation,
        ):
            totals = validator.validate_optibertneo_pair(verbose=False)

        self.assertEqual(allocation_devices, ["meta", "meta"])
        self.assertEqual(
            totals,
            {
                "real": validator.EXPECTED_TOTAL_PARAMETERS,
                "multispace": validator.EXPECTED_TOTAL_PARAMETERS,
            },
        )


class TestOptiBERTneoTrainingRecipe(unittest.TestCase):
    def run_launcher(self, variant: str, **overrides):
        environment = os.environ.copy()
        for name in ("SMOKE_TEST", "GLOBAL_SEQUENCES", "MICRO_BATCH"):
            environment.pop(name, None)
        environment.update(
            {
                "DRY_RUN": "1",
                "NUM_MACHINES": "2",
                "GPUS_PER_NODE": "4",
                "MACHINE_RANK": "0",
                "EXPECTED_NUM_MACHINES": "2",
                "EXPECTED_WORLD_SIZE": "8",
                "DATALOADER_WORKERS": "1",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            }
        )
        environment.update({key: str(value) for key, value in overrides.items()})
        return subprocess.run(
            ["bash", str(NEOBERT_ROOT / "jobs" / "optibertneo-1p3b.sh"), variant],
            cwd=NEOBERT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_submission_helper(self, dataset_path: Path, **overrides):
        environment = os.environ.copy()
        for name in (
            *UNSAFE_FULL_SUBMISSION_ENVIRONMENT,
            "DRY_RUN",
            "CONFIRM_FULL_SUBMISSION",
            "OPTIBERT_VARIANTS",
            "SBATCH_EXTRA_ARGS",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "H100_PARTITION": "test-h100",
                "H100_ACCOUNT": "test-account",
                "OPTIBERT_PYTHON": sys.executable,
                "OPTIBERT_DATASET": str(dataset_path),
                "RUNS_ROOT": str(dataset_path.parent / "runs"),
            }
        )
        environment.update({key: str(value) for key, value in overrides.items()})
        return subprocess.run(
            [
                "bash",
                str(NEOBERT_ROOT / "jobs" / "submit-optibertneo-pair.sh"),
                "both",
            ],
            cwd=NEOBERT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def make_minimal_submission_dataset(parent: Path) -> Path:
        dataset_path = parent / "prepared-dataset"
        (dataset_path / "tokenizer").mkdir(parents=True)
        for relative_path in (
            "dataset_info.json",
            "optibertneo_manifest.json",
            "tokenizer/tokenizer.json",
        ):
            (dataset_path / relative_path).touch()
        return dataset_path

    def test_paired_launcher_preserves_one_global_batch_and_token_budget(self):
        real = self.run_launcher("real")
        multispace = self.run_launcher("multispace")

        self.assertEqual(real.returncode, 0, real.stderr)
        self.assertEqual(multispace.returncode, 0, multispace.stderr)
        self.assertIn("model=optibertneo-198m", real.stdout)
        self.assertIn("micro_batch=32 gradient_accumulation=8", real.stdout)
        self.assertIn(
            "model=optibertneo-198m-multispace",
            multispace.stdout,
        )
        self.assertIn(
            "micro_batch=8 gradient_accumulation=32",
            multispace.stdout,
        )
        for output in (real.stdout, multispace.stdout):
            self.assertIn("global_batch=2048 sequences", output)
            self.assertIn("sequence_length=1024 steps=620 warmup_steps=500", output)
            self.assertIn("scheduled_tokens=1300234240", output)

    def test_production_launcher_rejects_a_global_batch_override(self):
        result = self.run_launcher("multispace", GLOBAL_SEQUENCES=1024)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fixes GLOBAL_SEQUENCES=2048", result.stderr)

    def test_paired_slurm_jobs_keep_variants_separate_and_resumable(self):
        job = (
            NEOBERT_ROOT
            / "jobs"
            / "slurm"
            / "optibertneo-paired-1p3b-2n8g.sbatch"
        ).read_text(encoding="utf-8")
        node = (
            NEOBERT_ROOT
            / "jobs"
            / "slurm"
            / "run-optibertneo-paired-node.sh"
        ).read_text(encoding="utf-8")
        submit = (
            NEOBERT_ROOT / "jobs" / "submit-optibertneo-pair.sh"
        ).read_text(encoding="utf-8")
        gpu_gate = (
            NEOBERT_ROOT
            / "jobs"
            / "slurm"
            / "test-optibertneo-pair-a100.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --nodes=2", job)
        self.assertIn("#SBATCH --gpus-per-task=h100:4", job)
        self.assertIn("MAX_TIME_SECONDS=${MAX_TIME_SECONDS:-13200}", job)
        self.assertIn("OPTIBERT_TRAINING_DEADLINE_EPOCH", job)
        launcher = (
            NEOBERT_ROOT / "jobs" / "optibertneo-1p3b.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("trainer.deadline_unix_seconds", launcher)
        self.assertIn("/$variant}", job)
        self.assertIn('--variant "$variant"', node)
        self.assertIn('validate_optibertneo_pair.py"', node)
        self.assertIn("dry_run=${DRY_RUN:-1}", submit)
        self.assertIn("CONFIRM_FULL_SUBMISSION", submit)
        self.assertIn("SMOKE_TEST=0", submit)
        self.assertIn("GLOBAL_SEQUENCES=2048", submit)
        self.assertIn("NUM_MACHINES=2", submit)
        self.assertIn("GPUS_PER_NODE=4", submit)
        self.assertIn("EXPECTED_NUM_MACHINES=2", submit)
        self.assertIn("EXPECTED_WORLD_SIZE=8", submit)
        self.assertIn("ACCELERATE_CONFIG=$project_root/conf/accelerate_ddp.yaml", submit)
        self.assertIn("variants=(real multispace)", submit)
        self.assertIn("$runs_root/real", submit)
        self.assertIn("$runs_root/multispace", submit)
        self.assertIn("${MULTISPACE_PRODUCTION_BATCH:-8}", gpu_gate)

    def test_submission_helper_defaults_to_dry_run_and_pins_full_run_shape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = self.make_minimal_submission_dataset(
                Path(temporary_directory)
            )
            result = self.run_submission_helper(dataset_path)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("sbatch --parsable"), 2)
        self.assertIn("OptiBERTneo real:", result.stdout)
        self.assertIn("OptiBERTneo multispace:", result.stdout)
        for expected in (
            "DRY_RUN=0",
            "SMOKE_TEST=0",
            "GLOBAL_SEQUENCES=2048",
            "NUM_MACHINES=2",
            "GPUS_PER_NODE=4",
            "EXPECTED_NUM_MACHINES=2",
            "EXPECTED_WORLD_SIZE=8",
            "ACCELERATE_CONFIG=",
        ):
            self.assertIn(expected, result.stdout)
        self.assertIn("MICRO_BATCH=32", result.stdout)
        self.assertIn("MICRO_BATCH=8", result.stdout)

    def test_live_submission_requires_explicit_affirmative_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = self.make_minimal_submission_dataset(
                Path(temporary_directory)
            )
            result = self.run_submission_helper(dataset_path, DRY_RUN=0)

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "DRY_RUN=0 and CONFIRM_FULL_SUBMISSION=YES",
            result.stderr,
        )
        self.assertNotIn("OptiBERTneo real: job", result.stdout)

    def test_submission_helper_rejects_inherited_run_shape_controls(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = self.make_minimal_submission_dataset(
                Path(temporary_directory)
            )
            for variable_name in UNSAFE_FULL_SUBMISSION_ENVIRONMENT:
                with self.subTest(variable_name=variable_name):
                    result = self.run_submission_helper(
                        dataset_path,
                        **{variable_name: "1"},
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(
                        f"Refusing inherited {variable_name}",
                        result.stderr,
                    )

    def test_1p3b_schedule_is_exact(self):
        sequence_length = 1_024
        global_sequences = 2_048
        optimizer_steps = 620
        tokens_per_step = sequence_length * global_sequences
        scheduled_tokens = tokens_per_step * optimizer_steps
        scheduled_rows = global_sequences * optimizer_steps

        self.assertEqual(tokens_per_step, 2_097_152)
        self.assertEqual(scheduled_rows, 1_269_760)
        self.assertEqual(scheduled_tokens, 1_300_234_240)

        scheduler = load_config("scheduler", "optibertneo_1p3b")
        trainer = load_config("trainer", "optibertneo_1p3b")
        self.assertEqual(scheduler["warmup_steps"], 500)
        self.assertEqual(scheduler["decay_steps"], optimizer_steps)
        self.assertEqual(scheduler["decay"], "cosine")
        self.assertEqual(scheduler["final_ratio"], 0.1)
        self.assertEqual(trainer["max_steps"], optimizer_steps)
        self.assertEqual(trainer["gradient_clipping"], 1)
        self.assertEqual(trainer["mixed_precision"], "bf16")
        self.assertTrue(trainer["tf32"])
        self.assertTrue(trainer["compile"])

    def test_fineweb_edu_roberta_and_mlm_contracts_are_pinned(self):
        dataset = load_config("dataset", "fineweb_edu")
        tokenizer = load_config("tokenizer", "roberta")
        collator = load_config("datacollator", "mlm_20")
        dataloader = load_config("dataloader", "optibertneo")

        self.assertEqual(dataset["column"], "text")
        self.assertEqual(dataset["approx_token_limit"], 1_600_000_000)
        self.assertEqual(dataset["expected_source_rows"], 9_672_101)
        self.assertEqual(dataset["pack_to_length"], 1_024)
        self.assertFalse(dataset["cross_document_attention"])
        self.assertIsNone(dataset["validation_fraction"])
        self.assertEqual(dataset["minimum_packed_rows"], 1_269_760)
        self.assertTrue(dataset["require_manifest"])
        self.assertEqual(
            dataset["train"],
            {
                "path": "HuggingFaceFW/fineweb-edu",
                "name": "sample-10BT",
                "split": "train",
                "revision": "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5",
            },
        )
        self.assertEqual(
            dataset["training_schedule"],
            {
                "optimizer_steps": 620,
                "global_sequences": 2_048,
                "required_token_positions": 1_300_234_240,
            },
        )
        self.assertNotIn("validation", dataset)

        self.assertEqual(
            tokenizer["pretrained_model_name_or_path"],
            "FacebookAI/roberta-base",
        )
        self.assertEqual(
            tokenizer["revision"],
            "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        )
        self.assertEqual(tokenizer["max_length"], 1_024)
        self.assertEqual(tokenizer["vocab_size"], 50_265)
        self.assertTrue(tokenizer["chunk_long_documents"])

        self.assertEqual(collator["mlm_probability"], 0.20)
        self.assertTrue(collator["mask_all"])
        self.assertEqual(collator["pad_to_multiple_of"], 8)
        self.assertTrue(dataloader["train"]["prepacked_sequences"])
        self.assertFalse(dataloader["train"]["pack_sequences"])

    def test_adamw_contract_matches_the_paper_recipe(self):
        optimizer = load_config("optimizer", "optibertneo")

        self.assertEqual(optimizer["name"], "AdamW")
        self.assertEqual(optimizer["hparams"]["lr"], 6e-4)
        self.assertEqual(optimizer["hparams"]["betas"], [0.9, 0.95])
        self.assertEqual(optimizer["hparams"]["eps"], 1e-8)
        self.assertEqual(optimizer["hparams"]["weight_decay"], 0.1)


if __name__ == "__main__":
    unittest.main()
