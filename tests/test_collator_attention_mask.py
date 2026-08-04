import unittest
from unittest import mock

import torch

import neobert.collator.collator as collator_module


class _FakeMLMCollator:
    output = None

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, batch):
        del batch
        return {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in self.output.items()
        }


class TestCollatorAttentionMask(unittest.TestCase):
    def _collate(self, attention_mask):
        _FakeMLMCollator.output = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": attention_mask,
        }
        with mock.patch.object(
            collator_module,
            "DataCollatorForLanguageModeling",
            _FakeMLMCollator,
        ):
            collate = collator_module.get_collator(
                tokenizer=object(),
                dtype=torch.bfloat16,
                mask_all=False,
            )
            return collate([{}])

    def test_all_valid_mask_is_omitted_instead_of_becoming_all_zero(self):
        batch = self._collate(torch.ones(1, 4, dtype=torch.int64))
        self.assertNotIn("attention_mask", batch)

    def test_padding_mask_keeps_additive_contract(self):
        batch = self._collate(torch.tensor([[1, 1, 0, 0]]))
        expected = torch.tensor(
            [[0.0, 0.0, float("-inf"), float("-inf")]],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(batch["attention_mask"], expected)


if __name__ == "__main__":
    unittest.main()
