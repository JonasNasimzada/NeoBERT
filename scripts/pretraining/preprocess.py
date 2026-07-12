import hydra
from omegaconf import DictConfig

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset

from neobert.tokenizer import get_tokenizer, tokenize


def select_approx_token_limit(dataset, token_limit):
    if token_limit is None:
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
                return dataset.select(range(row_count))
    raise ValueError(
        f"dataset contains only about {total_tokens:,} tokens, below the requested {token_limit:,}"
    )


def pack_tokenized_dataset(
    dataset,
    sequence_length,
    pad_token_id,
    bos_token_id,
    eos_token_id,
):
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "document_ids": Sequence(Value("int32")),
        }
    )

    def packed_rows():
        current_tokens = []
        current_document_ids = []
        segment_id = 0
        for row in dataset:
            input_ids = list(row["input_ids"])
            if input_ids and input_ids[0] == bos_token_id:
                input_ids = input_ids[1:]
            if input_ids and input_ids[-1] == eos_token_id:
                input_ids = input_ids[:-1]
            offset = 0
            while offset < len(input_ids):
                remaining = sequence_length - len(current_tokens)
                if remaining < 3:
                    current_tokens.extend([pad_token_id] * remaining)
                    current_document_ids.extend([-1] * remaining)
                    yield {
                        "input_ids": current_tokens,
                        "document_ids": current_document_ids,
                    }
                    current_tokens = []
                    current_document_ids = []
                    remaining = sequence_length

                take = min(remaining - 2, len(input_ids) - offset)
                current_tokens.append(bos_token_id)
                current_tokens.extend(input_ids[offset : offset + take])
                current_tokens.append(eos_token_id)
                current_document_ids.extend([segment_id] * (take + 2))
                offset += take
                segment_id += 1
                if len(current_tokens) == sequence_length:
                    yield {
                        "input_ids": current_tokens,
                        "document_ids": current_document_ids,
                    }
                    current_tokens = []
                    current_document_ids = []

        if current_tokens:
            padding = sequence_length - len(current_tokens)
            yield {
                "input_ids": current_tokens + [pad_token_id] * padding,
                "document_ids": current_document_ids + [-1] * padding,
            }

    print(f"Packing tokenized documents into fixed rows of {sequence_length:,} tokens")
    return Dataset.from_generator(packed_rows, features=features)


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
    dataset = select_approx_token_limit(
        dataset,
        cfg.dataset.get("approx_token_limit"),
    )

    print("Tokenizing dataset")
    dataset = tokenize(dataset, tokenizer, column_name=cfg.dataset.column, **cfg.tokenizer)
    if cfg.dataset.get("pack_to_length") is not None:
        dataset = pack_tokenized_dataset(
            dataset,
            cfg.dataset.pack_to_length,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        )
    minimum_packed_rows = cfg.dataset.get("minimum_packed_rows")
    if minimum_packed_rows is not None and len(dataset) < minimum_packed_rows:
        raise ValueError(
            f"preprocessing produced {len(dataset):,} rows, below the required {minimum_packed_rows:,}"
        )
    print(f"Prepared {len(dataset):,} training rows")

    # Save the tokenized dataset to disk
    print("Saving tokenized dataset")
    dataset.save_to_disk(cfg.dataset.path_to_disk, max_shard_size="1GB")


if __name__ == "__main__":
    preprocess()
