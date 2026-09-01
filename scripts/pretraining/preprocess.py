import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)

from neobert.tokenizer import get_tokenizer, tokenize


def select_approx_token_limit(dataset, token_limit, selection_metadata=None):
    """Select a source prefix and optionally record its exact size."""
    if token_limit is None:
        if selection_metadata is not None:
            selection_metadata["selected_source_rows"] = len(dataset)
            selection_metadata["selected_source_tokens"] = None
        return dataset
    if "token_count" not in dataset.column_names:
        raise ValueError("dataset.approx_token_limit requires a token_count column")

    total_tokens = 0
    row_count = 0
    for batch in dataset.iter(batch_size=10_000):
        for token_count in batch["token_count"]:
            total_tokens += token_count
            row_count += 1
            if total_tokens >= token_limit:
                print(f"Selected {row_count:,} documents with about {total_tokens:,} source tokens")
                if selection_metadata is not None:
                    selection_metadata["selected_source_rows"] = row_count
                    selection_metadata["selected_source_tokens"] = total_tokens
                return dataset.select(range(row_count))
    raise ValueError(
        f"dataset contains only about {total_tokens:,} tokens, below the requested {token_limit:,}"
    )


def validate_expected_source_rows(dataset, expected_rows=None):
    """Fail fast when a pinned full-corpus recipe resolves to the wrong size."""
    if expected_rows is None:
        return dataset
    if isinstance(dataset, DatasetDict):
        raise TypeError("source-row validation expects one resolved source split")
    expected_rows = int(expected_rows)
    if expected_rows <= 0:
        raise ValueError("dataset.expected_source_rows must be positive")
    actual_rows = len(dataset)
    if actual_rows != expected_rows:
        raise ValueError(
            "source dataset row count does not match the pinned full corpus: "
            f"got {actual_rows:,}, expected {expected_rows:,}"
        )
    print(f"Validated complete source dataset: {actual_rows:,} rows")
    return dataset


def create_train_validation_split(
    dataset,
    validation_fraction=None,
    seed=0,
):
    """Create a deterministic validation split before tokenization.

    Keeping the split at the source-row level prevents a tokenized segment from
    appearing in both train and validation. Existing recipes remain a single
    ``Dataset`` when ``validation_fraction`` is unset.
    """
    if validation_fraction is None:
        return dataset
    if isinstance(dataset, DatasetDict):
        raise TypeError("validation splitting expects a single source Dataset")

    validation_fraction = float(validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("dataset.validation_fraction must be between 0 and 1")

    split = dataset.train_test_split(
        test_size=validation_fraction,
        seed=int(seed),
        shuffle=True,
    )
    output = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
        }
    )
    print(
        "Created deterministic source-row split: "
        f"{len(output['train']):,} train, {len(output['validation']):,} validation "
        f"(seed={int(seed)})"
    )
    return output


def pack_tokenized_dataset(
    dataset,
    sequence_length,
    cache_dir=None,
    cross_document_attention=False,
):
    """Concatenate tokenized segments into full rows without padding.

    Each input row already contains the tokenizer's boundary special tokens. A
    segment may straddle two packed rows. By default, its tokens retain a
    document id so the model can block cross-document attention. When
    ``cross_document_attention`` is true, rows contain only ``input_ids``;
    this mask-free form is required by strict Flash SDPA. The single incomplete
    tail is discarded in either mode.
    """
    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    feature_columns = {"input_ids": Sequence(Value("int32"))}
    if not cross_document_attention:
        feature_columns["document_ids"] = Sequence(Value("int32"))
    features = Features(feature_columns)

    def packed_rows():
        current_tokens = []
        current_document_ids = []
        segment_id = 0
        for row in dataset:
            input_ids = list(row["input_ids"])
            offset = 0
            while offset < len(input_ids):
                remaining = sequence_length - len(current_tokens)
                take = min(remaining, len(input_ids) - offset)
                current_tokens.extend(input_ids[offset : offset + take])
                current_document_ids.extend([segment_id] * take)
                offset += take
                if len(current_tokens) == sequence_length:
                    packed_row = {"input_ids": current_tokens}
                    if not cross_document_attention:
                        packed_row["document_ids"] = current_document_ids
                    yield packed_row
                    current_tokens = []
                    current_document_ids = []
            segment_id += 1

        if current_tokens:
            print(
                f"Dropping the final {len(current_tokens):,}-token partial row "
                "to keep the packed dataset padding-free"
            )

    packing_mode = (
        "with cross-document attention"
        if cross_document_attention
        else "with document-boundary masks"
    )
    print(
        f"Packing tokenized documents into fixed rows of {sequence_length:,} tokens "
        f"{packing_mode}"
    )
    return Dataset.from_generator(
        packed_rows,
        features=features,
        cache_dir=cache_dir,
    )


def _dataset_splits(dataset):
    if isinstance(dataset, DatasetDict):
        return tuple(dataset.items())
    return (("train", dataset),)


def _split_manifest(dataset, sequence_length):
    token_positions = (
        len(dataset) * sequence_length
        if sequence_length is not None
        else None
    )
    return {
        "dataset_fingerprint": dataset._fingerprint,
        "rows": len(dataset),
        "tokens": token_positions,
        "packed_token_positions": token_positions,
        "columns": list(dataset.column_names),
    }


def save_preprocessed_dataset(
    dataset,
    tokenizer,
    cfg,
    *,
    source_rows=None,
    selected_source_rows=None,
    selected_source_tokens=None,
):
    output_path = Path(cfg.dataset.path_to_disk).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite prepared dataset: {output_path}. "
            "Choose a new dataset.path_to_disk or move the existing directory."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.tmp-",
            dir=output_path.parent,
        )
    )

    configured_sequence_length = cfg.dataset.get("pack_to_length")
    sequence_length = (
        int(configured_sequence_length)
        if configured_sequence_length is not None
        else None
    )
    cross_document_attention = bool(
        cfg.dataset.get("cross_document_attention", False)
    )
    configured_token_limit = cfg.dataset.get("approx_token_limit")
    split_manifests = {
        split_name: _split_manifest(split_dataset, sequence_length)
        for split_name, split_dataset in _dataset_splits(dataset)
    }
    primary_split_name = (
        str(cfg.dataset.get("train_split", "train"))
        if isinstance(dataset, DatasetDict)
        else "train"
    )
    if primary_split_name not in split_manifests:
        raise ValueError(
            f"configured training split {primary_split_name!r} is missing from prepared dataset"
        )
    primary_split = split_manifests[primary_split_name]
    manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": OmegaConf.to_container(cfg.dataset.train, resolve=True),
        "source_rows": int(source_rows) if source_rows is not None else None,
        "source_total_rows": (
            int(source_rows) if source_rows is not None else None
        ),
        "selected_source_rows": (
            int(selected_source_rows)
            if selected_source_rows is not None
            else None
        ),
        "selected_source_tokens": (
            int(selected_source_tokens)
            if selected_source_tokens is not None
            else None
        ),
        "source_token_limit": (
            int(configured_token_limit)
            if configured_token_limit is not None
            else None
        ),
        # Preserve the original top-level training fields for readers of the
        # single-split OptiBERTneo manifest while adding explicit split data.
        "dataset_fingerprint": primary_split["dataset_fingerprint"],
        "rows": primary_split["rows"],
        "sequence_length": sequence_length,
        "packed_token_positions": primary_split["packed_token_positions"],
        "splits": split_manifests,
        "packing": {
            "padding_free": sequence_length is not None,
            "cross_document_attention": (
                cross_document_attention
                if sequence_length is not None
                else None
            ),
            "document_ids": (
                not cross_document_attention
                if sequence_length is not None
                else None
            ),
            "document_id_padding_value": None,
        },
        "tokenizer": {
            "name": cfg.tokenizer.pretrained_model_name_or_path,
            "revision": cfg.tokenizer.get("revision"),
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "mask_token_id": tokenizer.mask_token_id,
        },
    }
    configured_training_schedule = cfg.dataset.get("training_schedule")
    if configured_training_schedule is not None:
        training_schedule = OmegaConf.to_container(
            configured_training_schedule,
            resolve=True,
        )
        if not isinstance(training_schedule, dict):
            raise TypeError("dataset.training_schedule must be a mapping")
        optimizer_steps = int(training_schedule["optimizer_steps"])
        global_sequences = int(training_schedule["global_sequences"])
        required_token_positions = int(
            training_schedule["required_token_positions"]
        )
        expected_token_positions = (
            optimizer_steps * global_sequences * int(sequence_length)
        )
        if min(optimizer_steps, global_sequences, required_token_positions) <= 0:
            raise ValueError("dataset.training_schedule values must be positive")
        if required_token_positions != expected_token_positions:
            raise ValueError(
                "dataset.training_schedule.required_token_positions must equal "
                "optimizer_steps * global_sequences * pack_to_length"
            )
        manifest["training_schedule"] = {
            "optimizer_steps": optimizer_steps,
            "global_sequences": global_sequences,
            "required_token_positions": required_token_positions,
        }
    elif cfg.dataset.name == "fineweb_edu":
        # Preserve the legacy OptiBERTneo paper manifest when its older
        # FineWeb-Edu/RoBERTa configuration is used.
        manifest["paper_schedule"] = {
            "optimizer_steps": 620,
            "global_sequences": 2048,
            "required_token_positions": 620 * 2048 * 1024,
        }

    try:
        print(f"Saving tokenized dataset atomically to {output_path}")
        dataset.save_to_disk(str(temporary_path), max_shard_size="1GB")
        tokenizer.save_pretrained(temporary_path / "tokenizer")
        (temporary_path / "optibertneo_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary_path, output_path)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


@hydra.main(version_base=None, config_path="../../conf", config_name="pretraining")
def preprocess(cfg: DictConfig):
    # Tokenizer
    tokenizer = get_tokenizer(**cfg.tokenizer)
    print(tokenizer)

    # Load and tokenize the dataset
    print("Loading dataset")
    if cfg.dataset.name == "wikibook":
        bookcorpus = load_dataset("bookcorpus", split="train")
        wiki = load_dataset("wikipedia", "20220301.en", split="train")
        wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

        assert bookcorpus.features.type == wiki.features.type
        dataset = concatenate_datasets([bookcorpus, wiki])
        dataset = dataset.shuffle(seed=0)
    else:
        dataset = load_dataset(**cfg.dataset.train)
    dataset = validate_expected_source_rows(
        dataset,
        cfg.dataset.get("expected_source_rows"),
    )
    source_rows = len(dataset)
    selection_metadata = {}
    dataset = select_approx_token_limit(
        dataset,
        cfg.dataset.get("approx_token_limit"),
        selection_metadata,
    )
    dataset = create_train_validation_split(
        dataset,
        cfg.dataset.get("validation_fraction"),
        cfg.dataset.get("validation_seed", cfg.get("seed", 0)),
    )

    print("Tokenizing dataset")
    if isinstance(dataset, DatasetDict):
        dataset = DatasetDict(
            {
                split_name: tokenize(
                    split_dataset,
                    tokenizer,
                    column_name=cfg.dataset.column,
                    **cfg.tokenizer,
                )
                for split_name, split_dataset in dataset.items()
            }
        )
    else:
        dataset = tokenize(
            dataset,
            tokenizer,
            column_name=cfg.dataset.column,
            **cfg.tokenizer,
        )
    if cfg.dataset.get("pack_to_length") is not None:
        cross_document_attention = bool(
            cfg.dataset.get("cross_document_attention", False)
        )
        if isinstance(dataset, DatasetDict):
            dataset = DatasetDict(
                {
                    split_name: pack_tokenized_dataset(
                        split_dataset,
                        cfg.dataset.pack_to_length,
                        cross_document_attention=cross_document_attention,
                    )
                    for split_name, split_dataset in dataset.items()
                }
            )
        else:
            dataset = pack_tokenized_dataset(
                dataset,
                cfg.dataset.pack_to_length,
                cross_document_attention=cross_document_attention,
            )
    minimum_packed_rows = cfg.dataset.get("minimum_packed_rows")
    train_split_name = str(cfg.dataset.get("train_split", "train"))
    train_dataset = (
        dataset[train_split_name]
        if isinstance(dataset, DatasetDict)
        else dataset
    )
    if minimum_packed_rows is not None and len(train_dataset) < minimum_packed_rows:
        raise ValueError(
            f"preprocessing produced {len(train_dataset):,} training rows, "
            f"below the required {minimum_packed_rows:,}"
        )
    for split_name, split_dataset in _dataset_splits(dataset):
        print(f"Prepared {len(split_dataset):,} {split_name} rows")

    save_preprocessed_dataset(
        dataset,
        tokenizer,
        cfg,
        source_rows=source_rows,
        selected_source_rows=selection_metadata.get("selected_source_rows"),
        selected_source_tokens=selection_metadata.get("selected_source_tokens"),
    )


if __name__ == "__main__":
    preprocess()
