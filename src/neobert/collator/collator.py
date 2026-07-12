import torch

from typing import Any, Optional, Tuple
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator


# Adapted from https://github.com/huggingface/transformers/blob/125de4164364420854d7fe537a9bd2fdaf7369d4/src/transformers/data/data_collator.py#L828
class CustomCollatorForMLM(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 100% MASK.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels


class DataCollatorWithPacking(DefaultDataCollator):
    """
    Data collator used for padding free approach, with sequence packing.
    """

    def __init__(self, sep_token_id, max_length, default_data_collator, **kwargs):
        super().__init__(**kwargs)
        self.sep_token_id = sep_token_id
        self.max_length = max_length
        self.default_data_collator = default_data_collator

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        packed_sequences = []
        packed_document_ids = []
        current_sequence = []
        current_document_ids = []

        for document_id, feature in enumerate(features):
            sequence = feature["input_ids"]
            current_sequence.extend(sequence)
            current_document_ids.extend([document_id] * len(sequence))
            while len(current_sequence) >= self.max_length:
                packed_sequences.append({"input_ids": current_sequence[: self.max_length]})
                packed_document_ids.append(current_document_ids[: self.max_length])
                current_sequence = current_sequence[self.max_length :]
                current_document_ids = current_document_ids[self.max_length :]

        if not packed_sequences and current_sequence:
            padding = self.max_length - len(current_sequence)
            packed_sequences.append(
                {"input_ids": current_sequence + [self.default_data_collator.tokenizer.pad_token_id] * padding}
            )
            packed_document_ids.append(current_document_ids + [-1] * padding)

        batch = self.default_data_collator(packed_sequences, return_tensors)
        batch.pop("attention_mask", None)
        batch["document_ids"] = torch.tensor(packed_document_ids, dtype=torch.int32)
        return batch


def get_collator(
    tokenizer,
    dtype: torch.dtype = torch.float32,
    mlm_probability: float = 0.15,
    pad_to_multiple_of: int = 8,
    mask_all: bool = False,
    pack_sequences: bool = False,
    prepacked_sequences: bool = False,
    max_length: int = 512,
):
    if pack_sequences and prepacked_sequences:
        raise ValueError("pack_sequences and prepacked_sequences are mutually exclusive")
    # No need to apply any padding if sequences are packed
    if pack_sequences:
        pad_to_multiple_of = None

    mlm_collator = (
        CustomCollatorForMLM(
            tokenizer=tokenizer,
            return_tensors="pt",
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        if mask_all
        else DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            return_tensors="pt",
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    )

    if pack_sequences:
        collator = DataCollatorWithPacking(
            sep_token_id=tokenizer.sep_token_id,
            max_length=max_length,
            default_data_collator=mlm_collator,
        )

        def collate_fn(batch):
            return collator(batch)

    elif prepacked_sequences:

        def collate_fn(batch):
            document_ids = torch.tensor(
                [feature["document_ids"] for feature in batch],
                dtype=torch.int32,
            )
            batch = mlm_collator(
                [{"input_ids": feature["input_ids"]} for feature in batch]
            )
            batch.pop("attention_mask", None)
            batch["document_ids"] = document_ids
            return batch

    else:

        def collate_fn(batch):
            batch = mlm_collator(batch)
            batch["attention_mask"] = torch.where(batch["attention_mask"] == 1, float(0.0), float("-inf")).type(dtype)
            return batch

    return collate_fn
