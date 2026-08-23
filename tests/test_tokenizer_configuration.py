"""Focused checks for tokenizer context metadata."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from neobert.tokenizer import get_tokenizer


class TestTokenizerConfiguration(unittest.TestCase):
    def test_configured_context_is_persisted_as_model_max_length(self):
        tokenizer = SimpleNamespace(model_max_length=512)
        with mock.patch(
            "neobert.tokenizer.tokenizer.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as from_pretrained:
            actual = get_tokenizer(
                pretrained_model_name_or_path="google-bert/bert-base-uncased",
                max_length=1024,
                revision="pinned-test-revision",
            )

        self.assertIs(actual, tokenizer)
        self.assertEqual(actual.model_max_length, 1024)
        self.assertEqual(
            from_pretrained.call_args.kwargs["model_max_length"],
            1024,
        )
        self.assertNotIn("max_length", from_pretrained.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
