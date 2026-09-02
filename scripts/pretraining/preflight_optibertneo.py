#!/usr/bin/env python3
"""Read-only preflight checks for the paired OptiBERTneo runs.

The training launcher invokes this program independently on every node.  It
does not initialize torch.distributed, contact the network, mutate the
dataset, or compile a kernel.  The module deliberately imports only the
standard library at import time so ``--config-only`` and the unit tests work
in a CPU-only environment without the training dependencies installed.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

SEQUENCE_LENGTH = 1_024
GLOBAL_SEQUENCES = 2_048
TRAINING_STEPS = 620
MINIMUM_PACKED_ROWS = 1_269_760
SCHEDULED_TOKENS = 1_300_234_240
PAPER_NON_EMBEDDING_PARAMETERS = 12 * 768**2 * 28
EXPECTED_NON_EMBEDDING_PARAMETERS = 198_225_408
EXPECTED_TOTAL_PARAMETERS = 236_828_928
REQUIRED_DATASET_COLUMNS = ("input_ids", "document_ids")
DATASET_SOURCE = {
    "path": "HuggingFaceFW/fineweb-edu",
    "name": "sample-10BT",
    "split": "train",
    "revision": "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5",
}
EXPECTED_SOURCE_ROWS = 9_672_101
TOKENIZER_IDENTITY = {
    "name": "FacebookAI/roberta-base",
    "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    "vocab_size": 50_265,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "pad_token_id": 1,
    "mask_token_id": 50_264,
}

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


@dataclass(frozen=True)
class Check:
    """One preflight observation."""

    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == FAIL


@dataclass
class Report:
    """Ordered collection of preflight observations."""

    checks: list[Check] = field(default_factory=list)

    def add(self, status: str, name: str, detail: str) -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    def passed(self, name: str, detail: str) -> None:
        self.add(PASS, name, detail)

    def failed(self, name: str, detail: str) -> None:
        self.add(FAIL, name, detail)

    def warned(self, name: str, detail: str) -> None:
        self.add(WARN, name, detail)

    def skipped(self, name: str, detail: str) -> None:
        self.add(SKIP, name, detail)

    def extend(self, checks: Iterable[Check]) -> None:
        self.checks.extend(checks)

    @property
    def failure_count(self) -> int:
        return sum(check.failed for check in self.checks)

    @property
    def exit_code(self) -> int:
        return int(bool(self.failure_count))

    def render(self) -> str:
        lines = [
            f"[{check.status:4}] {check.name}: {check.detail}"
            for check in self.checks
        ]
        counts = {
            status: sum(check.status == status for check in self.checks)
            for status in (PASS, FAIL, WARN, SKIP)
        }
        lines.append(
            "Summary: "
            + ", ".join(
                f"{counts[status]} {status.lower()}"
                for status in (PASS, FAIL, WARN, SKIP)
            )
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class Recipe:
    """Distributed batch and token arithmetic for one training run."""

    nodes: int = 2
    gpus_per_node: int = 4
    micro_batch: int = 32
    global_sequences: int = GLOBAL_SEQUENCES
    sequence_length: int = SEQUENCE_LENGTH
    steps: int = TRAINING_STEPS

    def __post_init__(self) -> None:
        for name in (
            "nodes",
            "gpus_per_node",
            "micro_batch",
            "global_sequences",
            "sequence_length",
            "steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.global_sequences % self.sequences_per_micro_step:
            raise ValueError(
                "global_sequences must be divisible by "
                "nodes * gpus_per_node * micro_batch"
            )

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def sequences_per_micro_step(self) -> int:
        return self.world_size * self.micro_batch

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.global_sequences // self.sequences_per_micro_step

    @property
    def scheduled_rows(self) -> int:
        return self.steps * self.global_sequences

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.global_sequences * self.sequence_length

    @property
    def scheduled_tokens(self) -> int:
        return self.scheduled_rows * self.sequence_length


@dataclass(frozen=True)
class ModelCounts:
    """Unique parameter counts for one tied-embedding MLM variant."""

    paper_target: int
    non_embedding: int
    total: int

    @property
    def relative_paper_difference(self) -> float:
        return (self.non_embedding - self.paper_target) / self.paper_target


@dataclass(frozen=True)
class DatasetSummary:
    """Metadata needed to validate a packed Hugging Face Dataset."""

    column_names: tuple[str, ...]
    num_rows: int
    min_lengths: Mapping[str, int]
    max_lengths: Mapping[str, int]
    element_types: Mapping[str, str]
    null_counts: Mapping[str, int]
    min_values: Mapping[str, int] = field(default_factory=dict)
    fingerprint: str | None = None
    fully_scanned: bool = True


@dataclass(frozen=True)
class TritonPins:
    """Triton expectations recorded by a sibling PyTorch checkout."""

    version: str | None
    commit: str | None


@dataclass(frozen=True)
class ImportRequirement:
    module: str
    distribution: str
    minimum: tuple[int, ...] | None = None
    maximum_exclusive: tuple[int, ...] | None = None


IMPORT_REQUIREMENTS = (
    # torch is source-built for this experiment, so only its minimum API level
    # is constrained here.  Commit/build checks below are more meaningful than
    # NeoBERT's historical wheel pin.  xFormers is intentionally optional for
    # the real recipe because it selects NativeSwiGLU.
    ImportRequirement("torch", "torch", (2, 5)),
    ImportRequirement("triton", "triton"),
    ImportRequirement("accelerate", "accelerate", (1, 1), (1, 2)),
    ImportRequirement("deepspeed", "deepspeed", (0, 15, 4), (0, 15, 5)),
    ImportRequirement("transformers", "transformers", (4, 46), (4, 47)),
    ImportRequirement("datasets", "datasets", (3, 1), (3, 2)),
    ImportRequirement("wandb", "wandb", (0, 18), (0, 19)),
    ImportRequirement("hydra", "hydra-core", (1, 3), (1, 4)),
    ImportRequirement("omegaconf", "omegaconf", (2, 3), (3,)),
    ImportRequirement("tqdm", "tqdm"),
    ImportRequirement("einops", "einops"),
)


EXPECTED_CONFIG_VALUES: Mapping[str, Mapping[str, Any]] = {
    "conf/model/optibertneo-198m.yaml": {
        "hidden_size": 768,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "intermediate_size": 3_072,
        "rope": True,
        "rms_norm": True,
        "embedding_rms_norm": True,
        "hidden_act": "swiglu",
        "fused_swiglu": False,
        "dropout": 0,
        "attention_dropout": 0,
        "norm_eps": 1e-5,
        "embedding_init_range": 0.02,
        "decoder_init_range": 0.02,
        "tie_word_embeddings": True,
        "lm_head_bias": False,
        "ngpt": False,
        "attention_space": "real",
        "attention_backend": "flex",
    },
    "conf/model/optibertneo-198m-multispace.yaml": {
        "hidden_size": 768,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "intermediate_size": 1_536,
        "rope": True,
        "rms_norm": True,
        "embedding_rms_norm": True,
        "hidden_act": "swiglu",
        "fused_swiglu": False,
        "dropout": 0,
        "attention_dropout": 0,
        "norm_eps": 1e-5,
        "embedding_init_range": 0.02,
        "decoder_init_range": 0.02,
        "tie_word_embeddings": True,
        "lm_head_bias": False,
        "ngpt": False,
        "attention_space": "multispace",
        "attention_backend": "flex",
        "multispace_cuda_streams": True,
    },
    "conf/tokenizer/roberta.yaml": {
        "pretrained_model_name_or_path": TOKENIZER_IDENTITY["name"],
        "revision": TOKENIZER_IDENTITY["revision"],
        "trust_remote_code": False,
        "max_length": SEQUENCE_LENGTH,
        "vocab_size": TOKENIZER_IDENTITY["vocab_size"],
        "truncation": True,
        "chunk_long_documents": True,
    },
    "conf/dataset/fineweb_edu.yaml": {
        "path_to_disk": "tokenized_datasets/fineweb_edu_roberta_1p6b",
        "approx_token_limit": 1_600_000_000,
        "expected_source_rows": EXPECTED_SOURCE_ROWS,
        "pack_to_length": SEQUENCE_LENGTH,
        "cross_document_attention": False,
        "validation_fraction": None,
        "minimum_packed_rows": MINIMUM_PACKED_ROWS,
        "require_manifest": True,
        "training_schedule.optimizer_steps": TRAINING_STEPS,
        "training_schedule.global_sequences": GLOBAL_SEQUENCES,
        "training_schedule.required_token_positions": SCHEDULED_TOKENS,
        "train.path": DATASET_SOURCE["path"],
        "train.name": DATASET_SOURCE["name"],
        "train.split": DATASET_SOURCE["split"],
        "train.revision": DATASET_SOURCE["revision"],
    },
    "conf/datacollator/mlm_20.yaml": {
        "mlm_probability": 0.20,
        "mask_all": True,
    },
    "conf/optimizer/optibertneo.yaml": {
        "name": "AdamW",
        "hparams.lr": 6e-4,
        "hparams.betas": [0.9, 0.95],
        "hparams.eps": 1e-8,
        "hparams.weight_decay": 0.1,
    },
    "conf/scheduler/optibertneo_1p3b.yaml": {
        "warmup_steps": 500,
        "decay_steps": TRAINING_STEPS,
        "decay": "cosine",
        "final_ratio": 0.1,
    },
    "conf/trainer/optibertneo_1p3b.yaml": {
        "tf32": True,
        "mixed_precision": "bf16",
        "resume": True,
        "max_steps": TRAINING_STEPS,
        "max_time_seconds": None,
        "deadline_unix_seconds": None,
        "minimum_optimizer_cycle_runway_seconds": 1_200,
        "gradient_clipping": 1,
        "compile": True,
        "find_unused_parameters": False,
        "model.save_at_end": True,
        "model.export_at_end": True,
    },
    "conf/dataloader/optibertneo.yaml": {
        "train.batch_size": 32,
        "train.shuffle": True,
        "train.pin_memory": True,
        "train.pack_sequences": False,
        "train.prepacked_sequences": True,
    },
}


def _release_tuple(version: str) -> tuple[int, ...]:
    """Return the numeric release prefix from a PEP 440-ish version."""

    match = re.match(r"\s*[vV]?(\d+(?:\.\d+)*)", str(version))
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _pad_version(version: Sequence[int], width: int) -> tuple[int, ...]:
    return tuple(version) + (0,) * max(0, width - len(version))


def version_in_range(
    version: str,
    minimum: tuple[int, ...] | None,
    maximum_exclusive: tuple[int, ...] | None,
) -> bool:
    """Compare release components without requiring ``packaging``."""

    release = _release_tuple(version)
    if not release:
        return False
    width = max(
        len(release),
        len(minimum or ()),
        len(maximum_exclusive or ()),
    )
    normalized = _pad_version(release, width)
    if minimum is not None and normalized < _pad_version(minimum, width):
        return False
    if (
        maximum_exclusive is not None
        and normalized >= _pad_version(maximum_exclusive, width)
    ):
        return False
    return True


def _version_constraint(requirement: ImportRequirement) -> str:
    constraints = []
    if requirement.minimum is not None:
        constraints.append(">=" + ".".join(map(str, requirement.minimum)))
    if requirement.maximum_exclusive is not None:
        constraints.append(
            "<" + ".".join(map(str, requirement.maximum_exclusive))
        )
    return ",".join(constraints) or "any version"


def _yaml_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return ast.literal_eval(value)
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
        value,
    ):
        return float(value)
    return value


def load_flat_yaml(path: Path) -> dict[str, Any]:
    """Load the simple mapping subset used by the shipped recipe.

    Avoiding a YAML dependency is intentional: configuration checks should
    still run before the environment has been installed.
    """

    values: dict[str, Any] = {}
    parents: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()
        if stripped.startswith("- "):
            raise ValueError(
                f"{path}:{line_number}: block lists are not supported"
            )
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected a mapping entry")
        key, raw_value = stripped.split(":", 1)
        while parents and indent <= parents[-1][0]:
            parents.pop()
        prefix = [parent for _, parent in parents]
        dotted_key = ".".join((*prefix, key.strip()))
        if raw_value.strip():
            values[dotted_key] = _yaml_scalar(raw_value)
        else:
            parents.append((indent, key.strip()))
    return values


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=0)
    return actual == expected


def calculate_real_model_counts(
    *,
    hidden_size: int = 768,
    num_hidden_layers: int = 28,
    intermediate_size: int = 3_072,
    vocab_size: int = 50_265,
    rms_norm: bool = True,
    embedding_rms_norm: bool = True,
    tie_word_embeddings: bool = True,
    lm_head_bias: bool = False,
) -> ModelCounts:
    """Calculate unique parameters without allocating the 198M model."""

    # NeoBERT rounds the 2/3 SwiGLU projection width to a multiple of eight.
    swiglu_width = math.ceil((2 * intermediate_size / 3) / 8) * 8
    attention_per_layer = 4 * hidden_size**2
    swiglu_per_layer = 3 * hidden_size * swiglu_width
    norm_per_layer = 2 * hidden_size if rms_norm else 4 * hidden_size
    non_embedding = num_hidden_layers * (
        attention_per_layer + swiglu_per_layer + norm_per_layer
    )

    # Embedding RMSNorm and the final encoder norm are not embeddings, and
    # therefore are included by inspect_optibertneo.py.
    if embedding_rms_norm:
        non_embedding += hidden_size
    non_embedding += hidden_size if rms_norm else 2 * hidden_size
    if not tie_word_embeddings:
        non_embedding += vocab_size * hidden_size
    if lm_head_bias:
        non_embedding += vocab_size

    embedding = vocab_size * hidden_size
    return ModelCounts(
        paper_target=12 * 768**2 * 28,
        non_embedding=non_embedding,
        total=embedding + non_embedding,
    )


def calculate_multispace_model_counts(
    *,
    hidden_size: int = 768,
    num_hidden_layers: int = 28,
    intermediate_size: int = 1_536,
    vocab_size: int = 50_265,
    rms_norm: bool = True,
    embedding_rms_norm: bool = True,
    tie_word_embeddings: bool = True,
    lm_head_bias: bool = False,
) -> ModelCounts:
    """Count the 4/4/4 multispace model's trainable real scalars."""

    swiglu_width = math.ceil((2 * intermediate_size / 3) / 8) * 8
    # H -> 6H packed two-component QKV plus one shared 2H -> H output.
    attention_per_layer = 8 * hidden_size**2
    swiglu_per_layer = 3 * hidden_size * swiglu_width
    norm_per_layer = 2 * hidden_size if rms_norm else 4 * hidden_size
    non_embedding = num_hidden_layers * (
        attention_per_layer + swiglu_per_layer + norm_per_layer
    )
    if embedding_rms_norm:
        non_embedding += hidden_size
    non_embedding += hidden_size if rms_norm else 2 * hidden_size
    if not tie_word_embeddings:
        non_embedding += vocab_size * hidden_size
    if lm_head_bias:
        non_embedding += vocab_size

    embedding = vocab_size * hidden_size
    return ModelCounts(
        paper_target=12 * 768**2 * 28,
        non_embedding=non_embedding,
        total=embedding + non_embedding,
    )


def _configuration_checks(project_root: Path) -> list[Check]:
    checks: list[Check] = []
    loaded: dict[str, dict[str, Any]] = {}
    for relative_path, expected_values in EXPECTED_CONFIG_VALUES.items():
        path = project_root / relative_path
        if not path.is_file():
            checks.append(Check(f"config.{relative_path}", FAIL, "file is missing"))
            continue
        try:
            actual_values = load_flat_yaml(path)
        except (OSError, ValueError, SyntaxError) as error:
            checks.append(
                Check(f"config.{relative_path}", FAIL, f"cannot parse: {error}")
            )
            continue
        loaded[relative_path] = actual_values
        mismatches = []
        for key, expected in expected_values.items():
            if key not in actual_values:
                mismatches.append(f"{key} is missing")
            elif not _same_value(actual_values[key], expected):
                mismatches.append(
                    f"{key}={actual_values[key]!r}, expected {expected!r}"
                )
        if mismatches:
            checks.append(
                Check(
                    f"config.{relative_path}",
                    FAIL,
                    "; ".join(mismatches),
                )
            )
        else:
            checks.append(
                Check(
                    f"config.{relative_path}",
                    PASS,
                    f"{len(expected_values)} required values match",
                )
            )

    tokenizer_values = loaded.get("conf/tokenizer/roberta.yaml")
    model_count_contracts = (
        (
            "real",
            "conf/model/optibertneo-198m.yaml",
            calculate_real_model_counts,
        ),
        (
            "multispace",
            "conf/model/optibertneo-198m-multispace.yaml",
            calculate_multispace_model_counts,
        ),
    )
    for variant, model_path, count_function in model_count_contracts:
        model_values = loaded.get(model_path)
        if model_values is None or tokenizer_values is None:
            continue
        try:
            counts = count_function(
                hidden_size=int(model_values["hidden_size"]),
                num_hidden_layers=int(model_values["num_hidden_layers"]),
                intermediate_size=int(model_values["intermediate_size"]),
                vocab_size=int(tokenizer_values["vocab_size"]),
                rms_norm=bool(model_values["rms_norm"]),
                embedding_rms_norm=bool(model_values["embedding_rms_norm"]),
                tie_word_embeddings=bool(model_values["tie_word_embeddings"]),
                lm_head_bias=bool(model_values["lm_head_bias"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            checks.append(
                Check(f"model.{variant}.parameter_count", FAIL, str(error))
            )
        else:
            if (
                counts.non_embedding != EXPECTED_NON_EMBEDDING_PARAMETERS
                or counts.total != EXPECTED_TOTAL_PARAMETERS
            ):
                checks.append(
                    Check(
                        f"model.{variant}.parameter_count",
                        FAIL,
                        f"calculated non-embedding={counts.non_embedding:,}, "
                        f"total={counts.total:,}",
                    )
                )
            elif abs(counts.relative_paper_difference) > 0.001:
                checks.append(
                    Check(
                        f"model.{variant}.parameter_count",
                        FAIL,
                        "non-embedding count differs from the paper target "
                        "by more than 0.1%",
                    )
                )
            else:
                checks.append(
                    Check(
                        f"model.{variant}.parameter_count",
                        PASS,
                        f"{variant}: {counts.non_embedding:,} non-embedding and "
                        f"{counts.total:,} total unique parameters "
                        f"({counts.relative_paper_difference:+.4%} vs paper)",
                    )
                )

    try:
        recipes = {
            "real": Recipe(micro_batch=32),
            "multispace": Recipe(micro_batch=8),
        }
    except ValueError as error:  # pragma: no cover - constants are immutable
        checks.append(Check("recipe.distributed_math", FAIL, str(error)))
    else:
        real_recipe = recipes["real"]
        multispace_recipe = recipes["multispace"]
        if (
            real_recipe.world_size != 8
            or real_recipe.gradient_accumulation_steps != 8
            or multispace_recipe.gradient_accumulation_steps != 32
            or any(
                recipe.scheduled_rows != MINIMUM_PACKED_ROWS
                or recipe.scheduled_tokens != SCHEDULED_TOKENS
                for recipe in recipes.values()
            )
        ):
            checks.append(
                Check(
                    "recipe.distributed_math",
                    FAIL,
                    "real accumulation="
                    f"{real_recipe.gradient_accumulation_steps}, multispace "
                    "accumulation="
                    f"{multispace_recipe.gradient_accumulation_steps}, rows="
                    f"{real_recipe.scheduled_rows:,}, tokens="
                    f"{real_recipe.scheduled_tokens:,}",
                )
            )
        else:
            checks.append(
                Check(
                    "recipe.distributed_math",
                    PASS,
                    "2 nodes x 4 GPUs: real microbatch 32/accumulation 8; "
                    "multispace microbatch 8/accumulation 32; "
                    f"{real_recipe.scheduled_tokens:,} scheduled tokens each",
                )
            )

    launcher = project_root / "jobs" / "optibertneo-1p3b.sh"
    if not launcher.is_file():
        checks.append(Check("config.launcher", FAIL, "launcher is missing"))
    else:
        text = launcher.read_text(encoding="utf-8")
        launcher_patterns = {
            "baseline real model": r"model_config=optibertneo-198m",
            "baseline microbatch": r"default_micro_batch=32",
            "multispace model": r"model_config=optibertneo-198m-multispace",
            "multispace microbatch": r"default_micro_batch=8",
            "global batch": r"global_sequences=\$\{GLOBAL_SEQUENCES:-2048\}",
            "sequence length": r"sequence_length=1024",
            "optimizer steps": r"training_steps=620",
            "world-size product": (
                r"world_size=\$\(\(num_machines \* gpus_per_node\)\)"
            ),
            "Slurm node count": r"SLURM_JOB_NUM_NODES",
            "Slurm GPUs per node": r"SLURM_GPUS_ON_NODE",
        }
        missing = [
            label
            for label, pattern in launcher_patterns.items()
            if re.search(pattern, text) is None
        ]
        if missing:
            checks.append(
                Check(
                    "config.launcher",
                    FAIL,
                    "missing recipe wiring: " + ", ".join(missing),
                )
            )
        else:
            checks.append(
                Check(
                    "config.launcher",
                    PASS,
                    "real and multispace distributed recipes are wired into launcher",
                )
            )
    return checks


def _module_origin(module: Any) -> str:
    origin = getattr(module, "__file__", None)
    if origin:
        return str(Path(origin).resolve())
    paths = getattr(module, "__path__", None)
    if paths:
        return ", ".join(str(Path(path).resolve()) for path in paths)
    return "<built-in or namespace>"


def _module_version(module: Any, distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        value = getattr(module, "__version__", None)
        return str(value) if value is not None else None


def _import_checks(
    requirements: Sequence[ImportRequirement] | None = None,
) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    modules: dict[str, Any] = {}
    for requirement in requirements or IMPORT_REQUIREMENTS:
        # DeepSpeed constructs Triton's autotune-cache manager while it is
        # imported.  When callers have not provided a cache directory, that
        # constructor otherwise tries to create ~/.triton/autotune.  Point it
        # at an already-existing temporary directory for this read-only
        # import probe, then restore the caller's environment exactly.
        temporary_triton_cache = (
            requirement.module == "deepspeed"
            and "TRITON_CACHE_DIR" not in os.environ
        )
        if temporary_triton_cache:
            os.environ["TRITON_CACHE_DIR"] = "/tmp"
        try:
            module = importlib.import_module(requirement.module)
        except Exception as error:
            checks.append(
                Check(
                    f"import.{requirement.module}",
                    FAIL,
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        finally:
            if temporary_triton_cache:
                os.environ.pop("TRITON_CACHE_DIR", None)
        modules[requirement.module] = module
        version = _module_version(module, requirement.distribution)
        if version is None:
            checks.append(
                Check(
                    f"import.{requirement.module}",
                    WARN,
                    f"imported from {_module_origin(module)}; version unavailable",
                )
            )
        elif not version_in_range(
            version,
            requirement.minimum,
            requirement.maximum_exclusive,
        ):
            checks.append(
                Check(
                    f"import.{requirement.module}",
                    FAIL,
                    f"version {version} does not satisfy "
                    f"{_version_constraint(requirement)} "
                    f"({_module_origin(module)})",
                )
            )
        else:
            checks.append(
                Check(
                    f"import.{requirement.module}",
                    PASS,
                    f"{version} ({_module_origin(module)})",
                )
            )
    return checks, modules


def _git_head(path: Path) -> str | None:
    try:
        process = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = process.stdout.strip().lower()
    return value if process.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None


def commits_match(actual: str | None, expected: str | None) -> bool | None:
    """Compare full or abbreviated Git hashes.

    ``None`` means that one side is not observable.
    """

    if not actual or not expected:
        return None
    actual = actual.lower()
    expected = expected.lower()
    if not (
        re.fullmatch(r"[0-9a-f]{7,40}", actual)
        and re.fullmatch(r"[0-9a-f]{7,40}", expected)
    ):
        return False
    return actual.startswith(expected) or expected.startswith(actual)


def _torch_source_checks(torch: Any, pytorch_source: Path | None) -> list[Check]:
    checks: list[Check] = []
    runtime_commit = getattr(getattr(torch, "version", None), "git_version", None)
    runtime_commit = str(runtime_commit).lower() if runtime_commit else None
    if pytorch_source is None:
        checks.append(
            Check(
                "torch.source_commit",
                WARN,
                "no sibling PyTorch source checkout was found",
            )
        )
        return checks
    source_commit = _git_head(pytorch_source)
    match = commits_match(runtime_commit, source_commit)
    if match is None:
        checks.append(
            Check(
                "torch.source_commit",
                WARN,
                f"cannot compare runtime={runtime_commit or 'unknown'} with "
                f"source={source_commit or 'unknown'}",
            )
        )
    elif not match:
        checks.append(
            Check(
                "torch.source_commit",
                FAIL,
                f"runtime {runtime_commit} does not match checkout "
                f"{source_commit}; rebuild/reinstall PyTorch",
            )
        )
    else:
        checks.append(
            Check(
                "torch.source_commit",
                PASS,
                f"runtime and {pytorch_source} are at {source_commit}",
            )
        )

    torch_origin = Path(torch.__file__).resolve()
    expected_package = (pytorch_source / "torch").resolve()
    try:
        torch_origin.relative_to(expected_package)
    except ValueError:
        checks.append(
            Check(
                "torch.source_origin",
                WARN,
                f"runtime imports {torch_origin}, outside {expected_package}",
            )
        )
    else:
        checks.append(
            Check(
                "torch.source_origin",
                PASS,
                f"runtime imports from {expected_package}",
            )
        )
    return checks


def read_triton_pins(pytorch_source: Path | None) -> TritonPins:
    if pytorch_source is None:
        return TritonPins(None, None)
    version_path = pytorch_source / ".ci" / "docker" / "triton_version.txt"
    commit_path = (
        pytorch_source
        / ".ci"
        / "docker"
        / "ci_commit_pins"
        / "triton.txt"
    )

    def first_value(path: Path) -> str | None:
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                value = raw_line.split("#", 1)[0].strip()
                if value:
                    return value
        except OSError:
            return None
        return None

    return TritonPins(first_value(version_path), first_value(commit_path))


def _commit_from_version(version: str | None) -> str | None:
    if not version:
        return None
    match = re.search(r"(?:\+|[-.])git([0-9a-f]{7,40})", version, re.I)
    return match.group(1).lower() if match else None


def _distribution_direct_url_commit(distribution: str) -> str | None:
    try:
        metadata = importlib.metadata.distribution(distribution)
        direct_url = metadata.read_text("direct_url.json")
        if not direct_url:
            return None
        value = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (
        importlib.metadata.PackageNotFoundError,
        json.JSONDecodeError,
        OSError,
    ):
        return None
    return str(value).lower() if value else None


def _module_checkout_commit(module: Any) -> str | None:
    origin = getattr(module, "__file__", None)
    if not origin:
        return None
    path = Path(origin).resolve().parent
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return _git_head(parent)
    return None


def _triton_checks(
    triton: Any,
    pytorch_source: Path | None,
) -> list[Check]:
    checks: list[Check] = []
    pins = read_triton_pins(pytorch_source)
    actual_version = _module_version(triton, "triton") or ""
    if pins.version is None:
        checks.append(
            Check(
                "triton.pin_version",
                WARN,
                "PyTorch triton_version.txt was not found",
            )
        )
    elif _release_tuple(actual_version) != _release_tuple(pins.version):
        checks.append(
            Check(
                "triton.pin_version",
                FAIL,
                f"runtime {actual_version or 'unknown'} != pinned {pins.version}",
            )
        )
    else:
        checks.append(
            Check(
                "triton.pin_version",
                PASS,
                f"runtime {actual_version} matches pinned {pins.version}",
            )
        )

    actual_commit = (
        _commit_from_version(actual_version)
        or getattr(triton, "__commit__", None)
        or getattr(triton, "__git_version__", None)
        or _distribution_direct_url_commit("triton")
        or _module_checkout_commit(triton)
    )
    actual_commit = str(actual_commit).lower() if actual_commit else None
    commit_match = commits_match(actual_commit, pins.commit)
    if pins.commit is None:
        checks.append(
            Check(
                "triton.pin_commit",
                WARN,
                "PyTorch Triton commit pin was not found",
            )
        )
    elif commit_match is None:
        checks.append(
            Check(
                "triton.pin_commit",
                FAIL,
                f"expected {pins.commit}; installed package does not expose "
                "its source commit, so the pin cannot be verified",
            )
        )
    elif not commit_match:
        checks.append(
            Check(
                "triton.pin_commit",
                FAIL,
                f"runtime {actual_commit} != pinned {pins.commit}",
            )
        )
    else:
        checks.append(
            Check(
                "triton.pin_commit",
                PASS,
                f"runtime commit {actual_commit} matches {pins.commit}",
            )
        )
    return checks


def _cuda_version_tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    text = str(value)
    if text.isdigit() and len(text) >= 3:
        number = int(text)
        # xFormers records CUDART_VERSION-style values as 1201 for 12.1.
        # Some extension APIs instead expose CUDA_VERSION-style 12060 for
        # 12.6, so accept both encodings.
        if number >= 10_000:
            return number // 1000, (number % 1000) // 10
        return number // 100, number % 100
    release = _release_tuple(text)
    if len(release) >= 2:
        return release[0], release[1]
    return None


def supports_cuda_capability(
    architectures: str | Iterable[str] | None,
    capability: tuple[int, int],
) -> bool:
    """Recognize common PyTorch/NVCC spellings for a CUDA capability."""

    if architectures is None:
        return False
    if isinstance(architectures, str):
        values = re.split(r"[\s,;]+", architectures)
    else:
        values = [str(value) for value in architectures]
    major, minor = capability
    compact = f"{major}{minor}"
    for value in values:
        normalized = value.strip().lower().replace("+ptx", "")
        if re.search(
            rf"(?:sm|compute)[_-]?{compact}a?(?:$|[^0-9])",
            normalized,
        ):
            return True
        if re.fullmatch(rf"{major}\.{minor}a?", normalized):
            return True
    return False


def supports_sm90(architectures: str | Iterable[str] | None) -> bool:
    """Recognize common PyTorch/NVCC spellings for Hopper code."""

    return supports_cuda_capability(architectures, (9, 0))


def _xformers_checks(
    xformers: Any,
    torch: Any,
    *,
    require_cuda_build: bool,
    require_sm90: bool,
) -> list[Check]:
    checks: list[Check] = []
    try:
        cpp_lib = importlib.import_module("xformers._cpp_lib")
        build_info = getattr(cpp_lib, "_build_metadata", None)
        load_error = getattr(cpp_lib, "_cpp_library_load_exception", None)
    except Exception as error:
        return [
            Check(
                "xformers.extension",
                FAIL,
                f"cannot inspect extension metadata: {type(error).__name__}: {error}",
            )
        ]

    has_cpp = bool(getattr(xformers, "_has_cpp_library", False))
    if has_cpp:
        checks.append(
            Check("xformers.extension", PASS, "C++/CUDA extension loaded")
        )
    elif require_cuda_build:
        checks.append(
            Check(
                "xformers.extension",
                FAIL,
                f"extension did not load: {load_error or 'unknown error'}",
            )
        )
    else:
        checks.append(
            Check(
                "xformers.extension",
                WARN,
                f"extension did not load: {load_error or 'unknown error'}",
            )
        )

    if build_info is None and load_error is not None:
        build_info = getattr(load_error, "build_info", None)
    if build_info is None:
        status = FAIL if require_cuda_build else WARN
        checks.append(
            Check(
                "xformers.build_metadata",
                status,
                "cpp_lib.json build metadata is unavailable",
            )
        )
        return checks

    build_torch = str(getattr(build_info, "torch_version", ""))
    runtime_torch = str(getattr(torch, "__version__", ""))
    if _release_tuple(build_torch)[:3] != _release_tuple(runtime_torch)[:3]:
        checks.append(
            Check(
                "xformers.build_torch",
                FAIL,
                f"built for torch {build_torch}, runtime is {runtime_torch}",
            )
        )
    else:
        checks.append(
            Check(
                "xformers.build_torch",
                PASS,
                f"build {build_torch}, runtime {runtime_torch}",
            )
        )

    build_python = str(getattr(build_info, "python_version", ""))
    runtime_python = platform.python_version()
    if _release_tuple(build_python)[:2] != _release_tuple(runtime_python)[:2]:
        checks.append(
            Check(
                "xformers.build_python",
                FAIL,
                f"built for Python {build_python}, runtime is {runtime_python}",
            )
        )
    else:
        checks.append(
            Check(
                "xformers.build_python",
                PASS,
                f"Python {runtime_python}",
            )
        )

    build_cuda = _cuda_version_tuple(getattr(build_info, "cuda_version", None))
    torch_cuda = _cuda_version_tuple(getattr(torch.version, "cuda", None))
    if build_cuda is None:
        status = FAIL if require_cuda_build else WARN
        checks.append(
            Check("xformers.build_cuda", status, "not built with CUDA")
        )
    elif torch_cuda is None:
        status = FAIL if require_cuda_build else WARN
        checks.append(
            Check(
                "xformers.build_cuda",
                status,
                f"xFormers CUDA {build_cuda[0]}.{build_cuda[1]}, "
                "but torch has no CUDA build metadata",
            )
        )
    elif build_cuda != torch_cuda:
        checks.append(
            Check(
                "xformers.build_cuda",
                FAIL,
                f"xFormers CUDA {build_cuda[0]}.{build_cuda[1]} != "
                f"torch CUDA {torch_cuda[0]}.{torch_cuda[1]}",
            )
        )
    else:
        checks.append(
            Check(
                "xformers.build_cuda",
                PASS,
                f"CUDA {build_cuda[0]}.{build_cuda[1]}",
            )
        )

    build_env = getattr(build_info, "build_env", {}) or {}
    arch_list = build_env.get("TORCH_CUDA_ARCH_LIST")
    if require_sm90 and not supports_sm90(arch_list):
        checks.append(
            Check(
                "xformers.build_sm90",
                FAIL,
                "TORCH_CUDA_ARCH_LIST does not prove SM90 support "
                f"({arch_list!r})",
            )
        )
    elif supports_sm90(arch_list):
        checks.append(
            Check(
                "xformers.build_sm90",
                PASS,
                f"TORCH_CUDA_ARCH_LIST={arch_list}",
            )
        )
    else:
        checks.append(
            Check(
                "xformers.build_sm90",
                WARN,
                f"SM90 is not recorded in build metadata ({arch_list!r})",
            )
        )

    try:
        ops = importlib.import_module("xformers.ops")
        swiglu = getattr(ops, "SwiGLU")
        if swiglu is None:
            raise AttributeError("SwiGLU is None")
    except Exception as error:
        checks.append(
            Check(
                "xformers.swiglu",
                FAIL,
                f"required NeoBERT SwiGLU is unavailable: {error}",
            )
        )
    else:
        checks.append(Check("xformers.swiglu", PASS, "SwiGLU is available"))
    return checks


def _optional_checks(checks: Iterable[Check]) -> list[Check]:
    """Turn optional-component failures into visible, non-blocking warnings."""

    return [
        Check(
            check.name,
            WARN if check.status == FAIL else check.status,
            check.detail,
        )
        for check in checks
    ]


def _attention_and_model_checks(
    torch: Any,
    project_root: Path,
    variant: str = "real",
) -> list[Check]:
    checks: list[Check] = []
    try:
        complex_attention = importlib.import_module("complex_attention")
        efficient_attention = getattr(complex_attention, "efficient_attention")
    except Exception as error:
        checks.append(
            Check(
                "import.complex_attention",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        )
        efficient_attention = None
    else:
        checks.append(
            Check(
                "import.complex_attention",
                PASS,
                _module_origin(complex_attention),
            )
        )

    try:
        functional = importlib.import_module("torch.nn.functional")
        sdpa = getattr(functional, "scaled_dot_product_attention")
        if not callable(sdpa):
            raise TypeError("scaled_dot_product_attention is not callable")
        query = torch.zeros((1, 1, 2, 8), dtype=torch.float32)
        output = (
            efficient_attention(query, query, query, backend="torch")
            if callable(efficient_attention)
            else sdpa(query, query, query)
        )
        if tuple(output.shape) != (1, 1, 2, 8):
            raise RuntimeError(f"unexpected output shape {tuple(output.shape)}")
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("native real attention returned non-finite values")
    except Exception as error:
        checks.append(
            Check(
                "attention.native_real",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        )
    else:
        checks.append(
            Check(
                "attention.native_real",
                PASS,
                "PyTorch SDPA and ComplexAttention's real torch path work on CPU",
            )
        )

    try:
        numpy_values = torch.tensor([1.0]).numpy().tolist()
        if numpy_values != [1.0]:
            raise RuntimeError(f"unexpected round-trip result: {numpy_values}")
    except Exception as error:
        checks.append(
            Check(
                "torch.numpy_bridge",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        )
    else:
        checks.append(
            Check(
                "torch.numpy_bridge",
                PASS,
                "Tensor.numpy() is available for downstream evaluation/export",
            )
        )

    try:
        flex_module = importlib.import_module(
            "torch.nn.attention.flex_attention"
        )
        flex_attention = getattr(flex_module, "flex_attention")
        create_block_mask = getattr(flex_module, "create_block_mask")
        if not callable(flex_attention) or not callable(create_block_mask):
            raise TypeError("FlexAttention entry points are not callable")
        if not callable(getattr(torch, "compile", None)):
            raise AttributeError("torch.compile is unavailable")
    except Exception as error:
        checks.append(
            Check(
                "attention.flex",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        )
    else:
        checks.append(
            Check(
                "attention.flex",
                PASS,
                "flex_attention, create_block_mask, and torch.compile exist",
            )
        )

    try:
        model_module = importlib.import_module("neobert.model")
        config_class = getattr(model_module, "NeoBERTConfig")
        model_class = getattr(model_module, "NeoBERTLMHead")
    except Exception as error:
        checks.append(
            Check(
                "import.neobert",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        )
        return checks
    checks.append(
        Check("import.neobert", PASS, _module_origin(model_module))
    )

    try:
        model_filenames = {
            "real": "optibertneo-198m.yaml",
            "multispace": "optibertneo-198m-multispace.yaml",
        }
        expected_space = "real" if variant == "real" else "multispace"
        model_values = load_flat_yaml(
            project_root / "conf" / "model" / model_filenames[variant]
        )
        config = config_class(
            vocab_size=50_265,
            pad_token_id=1,
            max_length=SEQUENCE_LENGTH,
            **model_values,
        )
        with torch.device("meta"):
            model = model_class(config)
        unique = {id(parameter): parameter for parameter in model.parameters()}
        embedding = model.model.encoder.weight
        non_embedding = sum(
            parameter.numel()
            for parameter in unique.values()
            if parameter is not embedding
        )
        total = sum(parameter.numel() for parameter in unique.values())
        spaces = set(config.attention_spaces)
        backends = set(config.attention_backends)
        if spaces != {expected_space} or backends != {"flex"}:
            raise RuntimeError(
                f"spaces={sorted(spaces)}, backends={sorted(backends)}"
            )
        if (
            non_embedding != EXPECTED_NON_EMBEDDING_PARAMETERS
            or total != EXPECTED_TOTAL_PARAMETERS
        ):
            raise RuntimeError(
                f"non-embedding={non_embedding:,}, total={total:,}"
            )
    except Exception as error:
        checks.append(
            Check(
                "model.runtime_count",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        )
    else:
        checks.append(
            Check(
                "model.runtime_count",
                PASS,
                f"all 28 layers are {expected_space}/Flex; {non_embedding:,} "
                f"non-embedding and {total:,} total unique parameters",
            )
        )
    return checks


def _gpu_checks(
    torch: Any,
    *,
    expected_gpus_per_node: int,
    required_gpu: str | None,
) -> list[Check]:
    checks: list[Check] = []
    device_contracts = {
        "a100": ("A100", (8, 0)),
        "h100": ("H100", (9, 0)),
    }
    device_contract = device_contracts.get(required_gpu)
    if getattr(torch.version, "cuda", None) is None:
        return [
            Check("cuda.build", FAIL, "PyTorch was not built with CUDA"),
            Check("cuda.runtime", FAIL, "CUDA is required for this invocation"),
        ]
    checks.append(
        Check("cuda.build", PASS, f"PyTorch CUDA {torch.version.cuda}")
    )
    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:
        return checks + [
            Check(
                "cuda.runtime",
                FAIL,
                f"CUDA availability check failed: {type(error).__name__}: {error}",
            )
        ]
    if not available:
        return checks + [
            Check("cuda.runtime", FAIL, "torch.cuda.is_available() is false")
        ]
    checks.append(Check("cuda.runtime", PASS, "CUDA runtime is available"))

    try:
        count = int(torch.cuda.device_count())
    except Exception as error:
        checks.append(
            Check("cuda.device_count", FAIL, f"cannot enumerate GPUs: {error}")
        )
        return checks
    if count != expected_gpus_per_node:
        checks.append(
            Check(
                "cuda.device_count",
                FAIL,
                f"{count} visible GPUs, expected {expected_gpus_per_node}",
            )
        )
    else:
        checks.append(
            Check(
                "cuda.device_count",
                PASS,
                f"{count} visible GPUs on this node",
            )
        )

    try:
        arch_list = tuple(torch.cuda.get_arch_list())
    except Exception as error:
        checks.append(
            Check("cuda.build_arch", FAIL, f"cannot query arch list: {error}")
        )
        arch_list = ()
    if device_contract is None:
        checks.append(
            Check(
                "cuda.build_arch",
                PASS if arch_list else WARN,
                ", ".join(arch_list) or "architecture list is empty",
            )
        )
    else:
        _, required_capability = device_contract
        architecture = f"sm_{required_capability[0]}{required_capability[1]}"
        supported = supports_cuda_capability(arch_list, required_capability)
        checks.append(
            Check(
                f"cuda.build_{architecture}",
                PASS if supported else FAIL,
                f"PyTorch {'includes' if supported else 'lacks'} {architecture} "
                f"({', '.join(arch_list)})",
            )
        )

    device_errors = []
    descriptions = []
    for index in range(count):
        try:
            name = str(torch.cuda.get_device_name(index))
            capability = tuple(torch.cuda.get_device_capability(index))
        except Exception as error:
            device_errors.append(f"cuda:{index}: {error}")
            continue
        descriptions.append(f"cuda:{index}={name} sm_{capability[0]}{capability[1]}")
        if device_contract is not None and (
            device_contract[0] not in name.upper()
            or capability != device_contract[1]
        ):
            device_errors.append(
                f"cuda:{index} is {name} with capability {capability}, "
                f"expected NVIDIA {device_contract[0]} "
                f"SM{device_contract[1][0]}{device_contract[1][1]}"
            )
    if device_errors:
        checks.append(Check("cuda.devices", FAIL, "; ".join(device_errors)))
    else:
        checks.append(
            Check(
                "cuda.devices",
                PASS if device_contract is not None else WARN,
                ", ".join(descriptions) or "no devices enumerated",
            )
        )

    try:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception as error:
        checks.append(
            Check("cuda.bf16", FAIL, f"BF16 query failed: {error}")
        )
    else:
        if bf16_supported:
            checks.append(Check("cuda.bf16", PASS, "BF16 is supported"))
        else:
            checks.append(
                Check("cuda.bf16", FAIL, "BF16 is not supported")
            )
    return checks


def _arrow_element_type(pa: Any, arrow_type: Any) -> str:
    while (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    ):
        arrow_type = arrow_type.value_type
    return str(arrow_type)


def summarize_huggingface_dataset(
    dataset: Any,
    *,
    saved_fingerprint: str | None = None,
) -> DatasetSummary:
    """Scan Arrow list offsets without materializing token arrays."""

    column_names = tuple(dataset.column_names)
    num_rows = len(dataset)
    table_holder = getattr(dataset, "data", None)
    table = getattr(table_holder, "table", None)
    if table is None:
        raise TypeError(
            "dataset does not expose its Arrow table; cannot prove fixed row lengths"
        )

    pa = importlib.import_module("pyarrow")
    pc = importlib.import_module("pyarrow.compute")
    min_lengths: dict[str, int] = {}
    max_lengths: dict[str, int] = {}
    element_types: dict[str, str] = {}
    null_counts: dict[str, int] = {}
    min_values: dict[str, int] = {}
    for column_name in REQUIRED_DATASET_COLUMNS:
        if column_name not in column_names:
            continue
        column = table.column(column_name)
        minima: list[int] = []
        maxima: list[int] = []
        null_count = 0
        value_minima: list[int] = []
        for chunk in column.chunks:
            null_count += int(chunk.null_count)
            element_types[column_name] = _arrow_element_type(pa, chunk.type)
            if pa.types.is_fixed_size_list(chunk.type):
                if len(chunk):
                    minima.append(int(chunk.type.list_size))
                    maxima.append(int(chunk.type.list_size))
            elif not (
                pa.types.is_list(chunk.type)
                or pa.types.is_large_list(chunk.type)
            ):
                raise TypeError(
                    f"{column_name} has non-list Arrow type {chunk.type}"
                )
            elif len(chunk):
                length_range = pc.min_max(pc.list_value_length(chunk)).as_py()
                if length_range["min"] is not None:
                    minima.append(int(length_range["min"]))
                    maxima.append(int(length_range["max"]))

            flattened = pc.list_flatten(chunk)
            null_count += int(flattened.null_count)
            # A minimum reduction over every Arrow values buffer proves the
            # absence of the historical -1 padding marker without converting
            # 1.3B Python integers or allocating a second token-sized array.
            if column_name == "document_ids" and len(chunk):
                minimum_value = pc.min(flattened).as_py()
                if minimum_value is not None:
                    value_minima.append(int(minimum_value))
        min_lengths[column_name] = min(minima) if minima else 0
        max_lengths[column_name] = max(maxima) if maxima else 0
        null_counts[column_name] = null_count
        if value_minima:
            min_values[column_name] = min(value_minima)
    return DatasetSummary(
        column_names=column_names,
        num_rows=num_rows,
        min_lengths=min_lengths,
        max_lengths=max_lengths,
        element_types=element_types,
        null_counts=null_counts,
        min_values=min_values,
        # load_from_disk intentionally derives a new in-memory fingerprint.
        # The original fingerprint recorded by save_to_disk lives in
        # state.json and is supplied by _dataset_checks.
        fingerprint=saved_fingerprint,
        fully_scanned=True,
    )


def validate_dataset_summary(
    summary: DatasetSummary,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    minimum_rows: int = MINIMUM_PACKED_ROWS,
) -> list[Check]:
    checks: list[Check] = []
    missing = [
        column
        for column in REQUIRED_DATASET_COLUMNS
        if column not in summary.column_names
    ]
    if missing:
        checks.append(
            Check(
                "dataset.schema",
                FAIL,
                "missing columns: " + ", ".join(missing),
            )
        )
    else:
        extra = sorted(set(summary.column_names) - set(REQUIRED_DATASET_COLUMNS))
        detail = "input_ids and document_ids are present"
        if extra:
            detail += f"; extra columns: {', '.join(extra)}"
        checks.append(
            Check("dataset.schema", WARN if extra else PASS, detail)
        )

    if summary.num_rows < minimum_rows:
        checks.append(
            Check(
                "dataset.rows",
                FAIL,
                f"{summary.num_rows:,} rows, need at least {minimum_rows:,}",
            )
        )
    else:
        checks.append(
            Check(
                "dataset.rows",
                PASS,
                f"{summary.num_rows:,} rows (minimum {minimum_rows:,})",
            )
        )

    for column in REQUIRED_DATASET_COLUMNS:
        if column not in summary.column_names:
            continue
        minimum = summary.min_lengths.get(column)
        maximum = summary.max_lengths.get(column)
        if minimum != sequence_length or maximum != sequence_length:
            checks.append(
                Check(
                    f"dataset.{column}.length",
                    FAIL,
                    f"min={minimum}, max={maximum}, expected "
                    f"{sequence_length}",
                )
            )
        else:
            checks.append(
                Check(
                    f"dataset.{column}.length",
                    PASS,
                    f"all {summary.num_rows:,} rows have length "
                    f"{sequence_length}",
                )
            )
        element_type = summary.element_types.get(column)
        if element_type != "int32":
            checks.append(
                Check(
                    f"dataset.{column}.dtype",
                    FAIL,
                    f"{element_type!r}, expected 'int32'",
                )
            )
        else:
            checks.append(
                Check(f"dataset.{column}.dtype", PASS, "int32")
            )
        null_count = summary.null_counts.get(column)
        if null_count:
            checks.append(
                Check(
                    f"dataset.{column}.nulls",
                    FAIL,
                    f"{null_count} null rows",
                )
            )
        else:
            checks.append(
                Check(f"dataset.{column}.nulls", PASS, "no null rows")
            )
        if column == "document_ids":
            minimum_value = summary.min_values.get(column)
            if minimum_value is None:
                checks.append(
                    Check(
                        "dataset.document_ids.padding",
                        FAIL,
                        "full-column minimum is unavailable; padding-free "
                        "packing cannot be proven",
                    )
                )
            elif minimum_value < 0:
                checks.append(
                    Check(
                        "dataset.document_ids.padding",
                        FAIL,
                        f"minimum document id is {minimum_value}; negative "
                        "values are padding markers",
                    )
                )
            else:
                checks.append(
                    Check(
                        "dataset.document_ids.padding",
                        PASS,
                        f"full Arrow scan found minimum document id "
                        f"{minimum_value}",
                    )
                )
    if not summary.fully_scanned:
        checks.append(
            Check(
                "dataset.full_scan",
                WARN,
                "only sampled rows were inspected",
            )
        )
    return checks


def validate_optibertneo_manifest(
    manifest: Any,
    summary: DatasetSummary,
) -> list[Check]:
    """Validate immutable provenance and packing facts recorded at save time."""

    if not isinstance(manifest, dict):
        return [
            Check(
                "dataset.manifest",
                FAIL,
                "top-level JSON value must be an object",
            )
        ]
    checks: list[Check] = []

    if manifest.get("format_version") != 1:
        checks.append(
            Check(
                "dataset.manifest.version",
                FAIL,
                f"{manifest.get('format_version')!r}, expected 1",
            )
        )
    else:
        checks.append(Check("dataset.manifest.version", PASS, "format 1"))

    expected_positions = summary.num_rows * SEQUENCE_LENGTH
    count_mismatches = []
    for key, expected in (
        ("rows", summary.num_rows),
        ("sequence_length", SEQUENCE_LENGTH),
        ("packed_token_positions", expected_positions),
    ):
        if manifest.get(key) != expected:
            count_mismatches.append(
                f"{key}={manifest.get(key)!r}, expected {expected!r}"
            )
    if expected_positions < SCHEDULED_TOKENS:
        count_mismatches.append(
            f"only {expected_positions:,} token positions, need "
            f"{SCHEDULED_TOKENS:,}"
        )
    if count_mismatches:
        checks.append(
            Check(
                "dataset.manifest.counts",
                FAIL,
                "; ".join(count_mismatches),
            )
        )
    else:
        checks.append(
            Check(
                "dataset.manifest.counts",
                PASS,
                f"{summary.num_rows:,} rows and {expected_positions:,} "
                "packed token positions",
            )
        )

    packing = manifest.get("packing")
    packing_mismatches = []
    if not isinstance(packing, dict):
        packing_mismatches.append("packing is not an object")
    else:
        expected_packing = {
            "padding_free": True,
            "cross_document_attention": False,
            "document_ids": True,
            "document_id_padding_value": None,
        }
        for key, expected in expected_packing.items():
            if key not in packing:
                packing_mismatches.append(f"packing.{key} is missing")
            elif packing[key] is not expected:
                packing_mismatches.append(
                    f"packing.{key}={packing[key]!r}, expected {expected!r}"
                )
    if packing_mismatches:
        checks.append(
            Check(
                "dataset.manifest.packing",
                FAIL,
                "; ".join(packing_mismatches),
            )
        )
    else:
        checks.append(
            Check(
                "dataset.manifest.packing",
                PASS,
                "padding-free with cross-document attention disabled",
            )
        )

    source = manifest.get("source")
    source_mismatches = []
    if not isinstance(source, dict):
        source_mismatches.append("source is not an object")
    else:
        for key, expected in DATASET_SOURCE.items():
            if source.get(key) != expected:
                source_mismatches.append(
                    f"source.{key}={source.get(key)!r}, expected {expected!r}"
                )
    if manifest.get("source_token_limit") != 1_600_000_000:
        source_mismatches.append(
            "source_token_limit="
            f"{manifest.get('source_token_limit')!r}, expected 1600000000"
        )
    if manifest.get("source_rows") != EXPECTED_SOURCE_ROWS:
        source_mismatches.append(
            f"source_rows={manifest.get('source_rows')!r}, expected "
            f"{EXPECTED_SOURCE_ROWS}"
        )
    if manifest.get("source_total_rows") != EXPECTED_SOURCE_ROWS:
        source_mismatches.append(
            f"source_total_rows={manifest.get('source_total_rows')!r}, expected "
            f"{EXPECTED_SOURCE_ROWS}"
        )
    selected_source_rows = manifest.get("selected_source_rows")
    if (
        not isinstance(selected_source_rows, int)
        or isinstance(selected_source_rows, bool)
        or not 0 < selected_source_rows <= EXPECTED_SOURCE_ROWS
    ):
        source_mismatches.append(
            "selected_source_rows must be a positive integer no larger than "
            f"{EXPECTED_SOURCE_ROWS}"
        )
    selected_source_tokens = manifest.get("selected_source_tokens")
    if (
        not isinstance(selected_source_tokens, int)
        or isinstance(selected_source_tokens, bool)
        or selected_source_tokens < 1_600_000_000
    ):
        source_mismatches.append(
            "selected_source_tokens must be an integer at least 1600000000"
        )
    fingerprint = manifest.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        source_mismatches.append("dataset_fingerprint is missing")
    elif (
        summary.fingerprint is not None
        and fingerprint != summary.fingerprint
    ):
        source_mismatches.append(
            f"dataset_fingerprint={fingerprint!r}, but loaded Arrow dataset "
            f"has {summary.fingerprint!r}"
        )
    if source_mismatches:
        checks.append(
            Check(
                "dataset.manifest.source",
                FAIL,
                "; ".join(source_mismatches),
            )
        )
    else:
        checks.append(
            Check(
                "dataset.manifest.source",
                PASS,
                "FineWeb-Edu sample-10BT commit and selected prefix are pinned",
            )
        )

    tokenizer = manifest.get("tokenizer")
    tokenizer_mismatches = []
    if not isinstance(tokenizer, dict):
        tokenizer_mismatches.append("tokenizer is not an object")
    else:
        for key, expected in TOKENIZER_IDENTITY.items():
            if tokenizer.get(key) != expected:
                tokenizer_mismatches.append(
                    f"tokenizer.{key}={tokenizer.get(key)!r}, "
                    f"expected {expected!r}"
                )
    if tokenizer_mismatches:
        checks.append(
            Check(
                "dataset.manifest.tokenizer",
                FAIL,
                "; ".join(tokenizer_mismatches),
            )
        )
    else:
        checks.append(
            Check(
                "dataset.manifest.tokenizer",
                PASS,
                "pinned RoBERTa revision, vocabulary, and special-token IDs",
            )
        )

    schedule = manifest.get("training_schedule")
    legacy_schedule = False
    if not isinstance(schedule, dict):
        schedule = manifest.get("paper_schedule")
        legacy_schedule = isinstance(schedule, dict)
    schedule_mismatches = []
    expected_schedule = {
        "optimizer_steps": TRAINING_STEPS,
        "global_sequences": GLOBAL_SEQUENCES,
        "required_token_positions": SCHEDULED_TOKENS,
    }
    if not isinstance(schedule, dict):
        schedule_mismatches.append("training_schedule is not an object")
    else:
        for key, expected in expected_schedule.items():
            if schedule.get(key) != expected:
                schedule_mismatches.append(
                    f"training_schedule.{key}={schedule.get(key)!r}, "
                    f"expected {expected!r}"
                )
    if schedule_mismatches:
        checks.append(
            Check(
                "dataset.manifest.schedule",
                FAIL,
                "; ".join(schedule_mismatches),
            )
        )
    else:
        checks.append(
            Check(
                "dataset.manifest.schedule",
                WARN if legacy_schedule else PASS,
                (
                    "legacy paper_schedule accepted; rebuild to record "
                    "training_schedule"
                    if legacy_schedule
                    else f"{TRAINING_STEPS} steps x {GLOBAL_SEQUENCES} sequences"
                ),
            )
        )
    return checks


def _local_tokenizer_checks(
    path: Path,
    transformers_module: Any | None,
) -> list[Check]:
    tokenizer_path = path / "tokenizer"
    if not (tokenizer_path / "tokenizer.json").is_file():
        return [
            Check(
                "dataset.tokenizer_files",
                FAIL,
                f"{tokenizer_path / 'tokenizer.json'} is missing",
            )
        ]
    if transformers_module is None:
        return [
            Check(
                "dataset.tokenizer_files",
                FAIL,
                "transformers did not import; saved tokenizer cannot be checked",
            )
        ]
    try:
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        return [
            Check(
                "dataset.tokenizer_files",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        ]
    actual = {
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
    }
    mismatches = [
        f"{key}={actual[key]!r}, expected {TOKENIZER_IDENTITY[key]!r}"
        for key in actual
        if actual[key] != TOKENIZER_IDENTITY[key]
    ]
    if mismatches:
        return [
            Check(
                "dataset.tokenizer_files",
                FAIL,
                "; ".join(mismatches),
            )
        ]
    return [
        Check(
            "dataset.tokenizer_files",
            PASS,
            "saved tokenizer matches pinned vocabulary and special-token IDs",
        )
    ]


def _dataset_checks(
    path: Path,
    datasets_module: Any | None,
    transformers_module: Any | None = None,
) -> list[Check]:
    if not path.exists():
        return [Check("dataset.path", FAIL, f"{path} does not exist")]
    if not path.is_dir():
        return [Check("dataset.path", FAIL, f"{path} is not a directory")]
    if datasets_module is None:
        return [
            Check(
                "dataset.load",
                FAIL,
                "the datasets package did not import",
            )
        ]
    checks = [Check("dataset.path", PASS, str(path.resolve()))]
    manifest_path = path / "optibertneo_manifest.json"
    manifest: Any = None
    manifest_was_parsed = False
    if not manifest_path.is_file():
        checks.append(
            Check(
                "dataset.manifest",
                FAIL,
                f"{manifest_path} is missing",
            )
        )
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_was_parsed = True
        except (OSError, json.JSONDecodeError) as error:
            checks.append(
                Check(
                    "dataset.manifest",
                    FAIL,
                    f"cannot parse: {error}",
                )
            )
    saved_fingerprint: str | None = None
    state_path = path / "state.json"
    try:
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        value = saved_state.get("_fingerprint")
        if isinstance(value, str) and value:
            saved_fingerprint = value
        else:
            checks.append(
                Check(
                    "dataset.state",
                    FAIL,
                    f"{state_path} has no _fingerprint",
                )
            )
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        checks.append(
            Check(
                "dataset.state",
                FAIL,
                f"cannot read {state_path}: {error}",
            )
        )
    try:
        dataset = datasets_module.load_from_disk(str(path))
    except Exception as error:
        checks.append(
            Check(
                "dataset.load",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        )
        return checks
    if isinstance(getattr(dataset, "column_names", None), dict):
        checks.append(
            Check(
                "dataset.load",
                FAIL,
                "expected a Dataset, got a DatasetDict",
            )
        )
        return checks
    try:
        summary = summarize_huggingface_dataset(
            dataset,
            saved_fingerprint=saved_fingerprint,
        )
    except Exception as error:
        checks.append(
            Check(
                "dataset.scan",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
        )
        return checks
    checks.append(
        Check(
            "dataset.scan",
            PASS,
            "read-only Arrow length scan completed",
        )
    )
    checks.extend(validate_dataset_summary(summary))
    if manifest_was_parsed:
        checks.extend(validate_optibertneo_manifest(manifest, summary))
    checks.extend(_local_tokenizer_checks(path, transformers_module))
    return checks


def _parse_positive_int(value: str, *, label: str) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def parse_slurm_gpu_count(value: str | None) -> int | None:
    """Parse ``4``, ``4(S:0-3)``, or ``gpu:h100:4``."""

    if value is None:
        return None
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    match = re.search(r"(?:^|:)(\d+)(?:\([^)]*\))?$", value)
    if match:
        return int(match.group(1))
    match = re.match(r"^(\d+)\(", value)
    return int(match.group(1)) if match else None


def _consistent_environment_integer(
    environment: Mapping[str, str],
    names: Sequence[str],
    parser: Any,
) -> tuple[int | None, str | None]:
    found: list[tuple[str, int]] = []
    invalid: list[str] = []
    for name in names:
        if name not in environment or environment[name] == "":
            continue
        value = parser(environment[name])
        if value is None:
            invalid.append(f"{name}={environment[name]!r}")
        else:
            found.append((name, value))
    if invalid:
        return None, "cannot parse " + ", ".join(invalid)
    if not found:
        return None, None
    distinct = {value for _, value in found}
    if len(distinct) != 1:
        return None, "conflicting values: " + ", ".join(
            f"{name}={value}" for name, value in found
        )
    return found[0][1], None


def validate_slurm_environment(
    environment: Mapping[str, str],
    *,
    expected_nodes: int = 2,
    expected_gpus_per_node: int = 4,
) -> list[Check]:
    checks: list[Check] = []
    nodes, nodes_error = _consistent_environment_integer(
        environment,
        ("NUM_MACHINES", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"),
        lambda value: _parse_positive_int(value, label="nodes"),
    )
    if nodes_error:
        checks.append(Check("slurm.nodes", FAIL, nodes_error))
    elif nodes is None:
        checks.append(
            Check("slurm.nodes", FAIL, "node-count variables are absent")
        )
    elif nodes != expected_nodes:
        checks.append(
            Check(
                "slurm.nodes",
                FAIL,
                f"{nodes} nodes, expected {expected_nodes}",
            )
        )
    else:
        checks.append(Check("slurm.nodes", PASS, f"{nodes} nodes"))

    gpus, gpus_error = _consistent_environment_integer(
        environment,
        ("GPUS_PER_NODE", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"),
        parse_slurm_gpu_count,
    )
    if gpus_error:
        checks.append(Check("slurm.gpus_per_node", FAIL, gpus_error))
    elif gpus is None:
        checks.append(
            Check(
                "slurm.gpus_per_node",
                FAIL,
                "GPU-per-node variables are absent",
            )
        )
    elif gpus != expected_gpus_per_node:
        checks.append(
            Check(
                "slurm.gpus_per_node",
                FAIL,
                f"{gpus} GPUs/node, expected {expected_gpus_per_node}",
            )
        )
    else:
        checks.append(
            Check("slurm.gpus_per_node", PASS, f"{gpus} GPUs/node")
        )

    expected_world = expected_nodes * expected_gpus_per_node
    computed_world = nodes * gpus if nodes is not None and gpus is not None else None
    explicit_world, world_error = _consistent_environment_integer(
        environment,
        ("WORLD_SIZE",),
        lambda value: _parse_positive_int(value, label="world size"),
    )
    if world_error:
        checks.append(Check("slurm.world_size", FAIL, world_error))
    elif computed_world is not None and computed_world != expected_world:
        checks.append(
            Check(
                "slurm.world_size",
                FAIL,
                f"nodes x GPUs/node = {computed_world}, expected "
                f"{expected_world}",
            )
        )
    elif explicit_world is not None and explicit_world != expected_world:
        checks.append(
            Check(
                "slurm.world_size",
                FAIL,
                f"WORLD_SIZE={explicit_world}, expected {expected_world}",
            )
        )
    elif (
        explicit_world is not None
        and computed_world is not None
        and explicit_world != computed_world
    ):
        checks.append(
            Check(
                "slurm.world_size",
                FAIL,
                f"WORLD_SIZE={explicit_world}, but nodes x GPUs/node="
                f"{computed_world}",
            )
        )
    else:
        checks.append(
            Check(
                "slurm.world_size",
                PASS,
                f"{explicit_world or computed_world or expected_world} processes",
            )
        )

    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_count = len([item for item in visible.split(",") if item.strip()])
        if visible_count != expected_gpus_per_node:
            checks.append(
                Check(
                    "slurm.cuda_visible_devices",
                    FAIL,
                    f"{visible_count} entries, expected "
                    f"{expected_gpus_per_node}: {visible}",
                )
            )
        else:
            checks.append(
                Check(
                    "slurm.cuda_visible_devices",
                    PASS,
                    f"{visible_count} visible devices",
                )
            )

    node_id = environment.get("SLURM_NODEID")
    if node_id is not None and nodes is not None:
        parsed_node_id = _parse_positive_int(node_id, label="node id")
        # SLURM_NODEID is zero-based, so zero needs separate handling.
        if node_id == "0":
            parsed_node_id = 0
        if parsed_node_id is None or not 0 <= parsed_node_id < nodes:
            checks.append(
                Check(
                    "slurm.node_id",
                    FAIL,
                    f"SLURM_NODEID={node_id!r} is outside [0, {nodes})",
                )
            )
        else:
            checks.append(
                Check("slurm.node_id", PASS, f"node rank {parsed_node_id}")
            )
    return checks


def should_validate_slurm(
    *,
    check_slurm: bool,
    skip_slurm: bool,
    require_gpu: bool,
    environment: Mapping[str, str],
) -> bool:
    """Decide whether topology is relevant for this invocation."""

    has_slurm_environment = any(
        name in environment
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NNODES",
        )
    )
    return bool(
        check_slurm
        or (has_slurm_environment and require_gpu and not skip_slurm)
    )


def _positive_argument(value: str) -> int:
    parsed = _parse_positive_int(value, label="argument")
    if parsed is None:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Read-only preflight for the paired OptiBERTneo "
            "1.3B-token run"
        )
    )
    parser.add_argument(
        "--variant",
        choices=("real", "multispace"),
        help=(
            "model whose runtime graph is checked; a dataset without a model "
            "or GPU requirement performs dataset-only validation"
        ),
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="check recipe files and pure arithmetic without importing training packages",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="validate a local Hugging Face dataset saved with save_to_disk",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="require CUDA, BF16, and the expected number of local GPUs",
    )
    gpu_type = parser.add_mutually_exclusive_group()
    gpu_type.add_argument(
        "--require-a100",
        action="store_true",
        help="also require every visible GPU and PyTorch to support A100/SM80",
    )
    gpu_type.add_argument(
        "--require-h100",
        action="store_true",
        help="also require every visible GPU and PyTorch to support H100/SM90",
    )
    parser.add_argument(
        "--expected-nodes",
        type=_positive_argument,
        default=2,
        help="expected Slurm node count (default: 2)",
    )
    parser.add_argument(
        "--expected-gpus-per-node",
        type=_positive_argument,
        default=4,
        help="expected visible GPUs on each node (default: 4)",
    )
    slurm_group = parser.add_mutually_exclusive_group()
    slurm_group.add_argument(
        "--check-slurm",
        action="store_true",
        help="require and validate Slurm allocation variables",
    )
    slurm_group.add_argument(
        "--skip-slurm",
        action="store_true",
        help="do not validate Slurm even if allocation variables are present",
    )
    parser.add_argument(
        "--pytorch-source",
        type=Path,
        help="PyTorch checkout to compare with the imported runtime",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help=argparse.SUPPRESS,
    )
    return parser


def run_preflight(args: argparse.Namespace) -> Report:
    report = Report()
    project_root = args.project_root.resolve()

    if sys.version_info < (3, 10):
        report.failed(
            "python.version",
            f"{platform.python_version()} is too old; Python >=3.10 is required",
        )
    else:
        report.passed(
            "python.version",
            f"{platform.python_version()} ({sys.executable})",
        )
    report.extend(_configuration_checks(project_root))

    if args.config_only:
        report.skipped(
            "runtime",
            "training imports, dataset, CUDA, xFormers build, and Slurm checks "
            "were disabled by --config-only",
        )
        return report

    dataset_only = (
        args.dataset is not None
        and args.variant is None
        and not args.require_gpu
        and not args.require_a100
        and not args.require_h100
        and not args.check_slurm
        and args.pytorch_source is None
    )
    if dataset_only:
        requirements = tuple(
            requirement
            for requirement in IMPORT_REQUIREMENTS
            if requirement.module in {"datasets", "transformers"}
        )
        import_checks, modules = _import_checks(requirements)
        report.extend(import_checks)
        report.extend(
            _dataset_checks(
                args.dataset.resolve(),
                modules.get("datasets"),
                modules.get("transformers"),
            )
        )
        report.skipped(
            "runtime",
            "PyTorch, Triton, CUDA, model, and Slurm checks are separate from "
            "dataset-only validation",
        )
        return report

    import_checks, modules = _import_checks()
    report.extend(import_checks)
    torch = modules.get("torch")
    triton = modules.get("triton")
    try:
        xformers = importlib.import_module("xformers")
    except Exception as error:
        xformers = None
        report.warned(
            "import.xformers",
            "optional because fused_swiglu=false; "
            f"{type(error).__name__}: {error}",
        )
    else:
        report.passed(
            "import.xformers",
            "optional package found at " + _module_origin(xformers),
        )

    pytorch_source = args.pytorch_source
    if pytorch_source is None:
        candidate = project_root.parent / "pytorch"
        pytorch_source = candidate if candidate.exists() else None
    elif not pytorch_source.exists():
        report.failed(
            "torch.source_path",
            f"{pytorch_source} does not exist",
        )
        pytorch_source = None

    if torch is not None:
        report.extend(_torch_source_checks(torch, pytorch_source))
        report.extend(
            _attention_and_model_checks(
                torch,
                project_root,
                args.variant or "real",
            )
        )
        if args.variant == "multispace":
            report.warned(
                "attention.multispace_flex_memory",
                "the dual-number tangent path currently materializes a dense "
                "B x H x S x S mask/JVP; the configured microbatch 8 passed a "
                "full 28-layer, sequence-length-1024 A40 smoke, but must still "
                "pass the target-allocation gate before training",
            )
    else:
        report.skipped("torch.runtime", "torch did not import")

    if triton is not None:
        report.extend(_triton_checks(triton, pytorch_source))
    else:
        report.skipped("triton.pins", "triton did not import")

    required_gpu = (
        "a100" if args.require_a100 else "h100" if args.require_h100 else None
    )
    require_gpu = bool(args.require_gpu or required_gpu)
    if xformers is not None and torch is not None:
        report.extend(
            _optional_checks(
                _xformers_checks(
                    xformers,
                    torch,
                    require_cuda_build=False,
                    require_sm90=False,
                )
            )
        )
    else:
        report.skipped(
            "xformers.build",
            "optional xFormers metadata unavailable; paired models use "
            "fused_swiglu=false",
        )

    if args.dataset is not None:
        report.extend(
            _dataset_checks(
                args.dataset.resolve(),
                modules.get("datasets"),
                modules.get("transformers"),
            )
        )
    else:
        report.skipped(
            "dataset",
            "no --dataset path was supplied",
        )

    if require_gpu:
        if torch is None:
            report.failed("cuda.runtime", "torch did not import")
        else:
            report.extend(
                _gpu_checks(
                    torch,
                    expected_gpus_per_node=args.expected_gpus_per_node,
                    required_gpu=required_gpu,
                )
            )
    else:
        report.skipped(
            "cuda.runtime",
            "GPU checks require --require-gpu, --require-a100, or --require-h100",
        )

    # Dataset preparation also runs under Slurm, but intentionally uses a
    # single CPU node.  Infer the 2x4 topology check only for a requested GPU
    # preflight; --check-slurm remains available to force it explicitly.
    if should_validate_slurm(
        check_slurm=args.check_slurm,
        skip_slurm=args.skip_slurm,
        require_gpu=require_gpu,
        environment=os.environ,
    ):
        report.extend(
            validate_slurm_environment(
                os.environ,
                expected_nodes=args.expected_nodes,
                expected_gpus_per_node=args.expected_gpus_per_node,
            )
        )
    else:
        report.skipped(
            "slurm.environment",
            "not in a Slurm allocation"
            if not args.skip_slurm
            else "disabled by --skip-slurm",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config_only and (
        args.dataset is not None
        or args.require_gpu
        or args.require_a100
        or args.require_h100
        or args.check_slurm
    ):
        parser.error(
            "--config-only cannot be combined with --dataset, GPU requirements, "
            "or --check-slurm"
        )
    report = run_preflight(args)
    print(report.render())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
