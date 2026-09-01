"""Static contracts for the parameter-matched real-MHA FineWeb-Edu run."""

from __future__ import annotations

import ast
import subprocess
import unittest
from pathlib import Path

from omegaconf import OmegaConf


NEOBERT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 30_522
EXPECTED_BLOCK_PARAMETERS = 8_504_832
EXPECTED_TRAINABLE_PARAMETERS = 99_985_152


def _load_model_config(filename: str) -> dict:
    values = OmegaConf.to_container(
        OmegaConf.load(NEOBERT_ROOT / "conf" / "model" / filename),
        resolve=True,
    )
    if not isinstance(values, dict):
        raise TypeError(f"{filename} must contain a mapping")
    return values


class TestRealFineWebTraining(unittest.TestCase):
    def test_real_and_multispace_configs_are_exactly_parameter_matched(self):
        real = _load_model_config("attention-ablation-real-100m.yaml")
        multispace = _load_model_config("attention-ablation-multispace.yaml")

        for key, expected in {
            "hidden_size": 768,
            "num_hidden_layers": 9,
            "num_attention_heads": 12,
            "rope": True,
            "rms_norm": True,
            "embedding_rms_norm": False,
            "hidden_act": "gelu",
            "dropout": 0,
            "attention_dropout": 0,
            "tie_word_embeddings": True,
            "lm_head_bias": False,
            "ngpt": False,
            "attention_backend": "flash",
        }.items():
            with self.subTest(config_key=key):
                self.assertEqual(real[key], expected)
                self.assertEqual(multispace[key], expected)

        self.assertEqual(real["attention_space"], "real")
        self.assertEqual(real["intermediate_size"], 4_000)
        self.assertEqual(multispace["attention_space"], "multispace")
        self.assertEqual(multispace["intermediate_size"], 2_464)
        self.assertTrue(multispace["multispace_cuda_streams"])

        hidden_size = real["hidden_size"]
        num_layers = real["num_hidden_layers"]
        num_heads = real["num_attention_heads"]
        self.assertEqual(hidden_size % num_heads, 0)
        self.assertEqual(hidden_size // num_heads, 64)

        # Real MHA has H->3H packed QKV and H->H output projections.
        real_attention = 4 * hidden_size**2
        real_ffn = 2 * hidden_size * real["intermediate_size"]
        norm_parameters = 2 * hidden_size
        real_block = real_attention + real_ffn + norm_parameters

        # Multispace has H->6H packed two-component QKV and 2H->H output.
        multispace_attention = 8 * hidden_size**2
        multispace_ffn = (
            2 * hidden_size * multispace["intermediate_size"]
        )
        multispace_block = (
            multispace_attention + multispace_ffn + norm_parameters
        )

        shared_parameters = VOCAB_SIZE * hidden_size + hidden_size
        real_total = shared_parameters + num_layers * real_block
        multispace_total = shared_parameters + num_layers * multispace_block

        self.assertEqual(real_attention, 2_359_296)
        self.assertEqual(multispace_attention, 4_718_592)
        self.assertEqual(real_block, EXPECTED_BLOCK_PARAMETERS)
        self.assertEqual(multispace_block, EXPECTED_BLOCK_PARAMETERS)
        self.assertEqual(real_total, EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(multispace_total, EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(real_total, multispace_total)

    def test_pair_validator_pins_the_same_static_contract(self):
        validator_path = (
            NEOBERT_ROOT
            / "scripts"
            / "attention_ablation"
            / "validate_100m_pair.py"
        )
        source = validator_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(validator_path))
        constants = {}
        for statement in tree.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                try:
                    constants[statement.targets[0].id] = ast.literal_eval(
                        statement.value
                    )
                except (ValueError, TypeError):
                    pass

        self.assertEqual(constants["VOCAB_SIZE"], VOCAB_SIZE)
        self.assertEqual(constants["HIDDEN_SIZE"], 768)
        self.assertEqual(constants["NUM_HIDDEN_LAYERS"], 9)
        self.assertEqual(constants["NUM_ATTENTION_HEADS"], 12)
        self.assertEqual(
            constants["EXPECTED_BLOCK_PARAMETERS"],
            EXPECTED_BLOCK_PARAMETERS,
        )
        self.assertEqual(
            constants["EXPECTED_TRAINABLE_PARAMETERS"],
            EXPECTED_TRAINABLE_PARAMETERS,
        )
        self.assertIn('config_filename="attention-ablation-real-100m.yaml"', source)
        self.assertIn('config_filename="attention-ablation-multispace.yaml"', source)
        self.assertIn("if len(set(counts.values())) != 1:", source)

    def test_jobs_are_syntax_valid_and_fixed_to_the_paired_recipe(self):
        jobs = NEOBERT_ROOT / "jobs" / "real_fineweb"
        train_path = jobs / "train.sbatch"
        submit_path = jobs / "submit.sh"
        train = train_path.read_text(encoding="utf-8")
        submit = submit_path.read_text(encoding="utf-8")

        for path in (train_path, submit_path):
            subprocess.run(["bash", "-n", str(path)], check=True)

        self.assertIn("#SBATCH --gpus=A100:1", train)
        self.assertIn('ATTENTION_VARIANT="real-flash"', train)
        self.assertIn('ATTENTION_SPACE="real"', train)
        self.assertIn('ATTENTION_BACKEND="flash"', train)
        self.assertIn('MODEL_CONFIG="attention-ablation-real-100m"', train)
        self.assertIn("scripts/attention_ablation/validate_100m_pair.py", train)
        self.assertIn("--space real", train)
        self.assertIn("--backend flash", train)
        self.assertIn("--require-a100", train)
        self.assertIn("dataset=fineweb_edu_google_1024", train)
        self.assertIn("tokenizer=google-1024", train)
        self.assertIn("model=attention-ablation-real-100m", train)
        self.assertIn('MAX_STEPS="${MAX_STEPS:-84000}"', train)
        self.assertIn('MICRO_BATCH="${MICRO_BATCH:-4}"', train)
        self.assertIn('GRAD_ACCUM="${GRAD_ACCUM:-4}"', train)
        self.assertIn('MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-53640}"', train)
        self.assertNotIn("resolve_attention_variant", train)

        self.assertIn("../multispace_fineweb/prepare_data.sbatch", submit)
        self.assertIn('RUNS_ROOT="$(realpath -m "${RUNS_ROOT:-$neobert_root/logs/real_fineweb}")"', submit)
        self.assertIn(
            'EXPERIMENT_ID="${EXPERIMENT_ID:-fineweb-edu-s1024-real-100m-v1}"',
            submit,
        )
        self.assertIn('TRAIN_SEGMENTS="${TRAIN_SEGMENTS:-5}"', submit)
        self.assertIn('PREP_JOB_ID="${PREP_JOB_ID:-}"', submit)
        self.assertIn('SKIP_PREP="${SKIP_PREP:-0}"', submit)
        self.assertIn("--gpus=A100:1", submit)
        self.assertIn('dependency_args=(--dependency="afterok:$previous_job_id")', submit)
        self.assertIn("MAX_STEPS=84000", submit)
        self.assertIn("MICRO_BATCH=4", submit)
        self.assertIn("GRAD_ACCUM=4", submit)
        self.assertIn("WARMUP_STEPS=5469", submit)
        self.assertIn("CHECKPOINT_STEPS=14000", submit)
        self.assertIn("EVAL_STEPS=14000", submit)
        self.assertIn("LOG_INTERVAL=840", submit)
        self.assertIn("MAX_TIME_SECONDS=53640", submit)
        self.assertNotIn("benchmark", submit)


if __name__ == "__main__":
    unittest.main()
