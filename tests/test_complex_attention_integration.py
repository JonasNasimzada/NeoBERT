import math
import unittest
from unittest import mock

import torch
from torch import nn

import neobert.model.model as model_module
from neobert.model import NeoBERT, NeoBERTConfig


class RecordingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.arguments = None

    def forward(
        self,
        x,
        pad_mask,
        freqs_cis,
        key_padding_mask,
        block_mask,
        dual_attention_mask,
    ):
        self.arguments = (
            pad_mask,
            key_padding_mask,
            block_mask,
            dual_attention_mask,
        )
        return x


class RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_mask = None

    def forward(
        self,
        x,
        attn_mask,
        key_padding_mask,
        freqs_cis,
        block_mask,
    ):
        self.block_mask = block_mask
        return torch.zeros_like(x)


class TestComplexAttentionIntegration(unittest.TestCase):
    def test_padding_bias_stays_compact(self):
        pad_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        bias, key_padding_mask = model_module._prepare_attention_masks(
            pad_mask,
            num_heads=4,
            seq_len=4,
        )

        self.assertEqual(bias.shape, (2, 1, 1, 4))
        torch.testing.assert_close(key_padding_mask, pad_mask.logical_not())

    def test_ambiguous_float_padding_masks_are_rejected(self):
        for pad_mask in (
            torch.tensor([[0.0, 1.5, float("-inf"), 0.0]]),
            torch.tensor([[0.0, -2.0, float("-inf"), 0.0]]),
        ):
            with self.subTest(pad_mask=pad_mask):
                with self.assertRaisesRegex(ValueError, "only 0/-inf"):
                    model_module._prepare_attention_masks(
                        pad_mask,
                        num_heads=2,
                        seq_len=4,
                    )

    def test_additive_padding_mask_has_backend_independent_semantics(self):
        pad_mask = torch.tensor([[0.0, 0.0, float("-inf"), float("-inf")]])
        bias, key_padding_mask = model_module._prepare_attention_masks(
            pad_mask,
            num_heads=2,
            seq_len=4,
        )

        torch.testing.assert_close(bias, pad_mask[:, None, None, :])
        torch.testing.assert_close(
            key_padding_mask,
            torch.tensor([[False, False, True, True]]),
        )

    def test_padding_flex_mask_matches_key_padding_semantics(self):
        document_ids = torch.tensor([[0, 0, -1, -1]], dtype=torch.int32)
        with mock.patch(
            "torch.nn.attention.flex_attention.create_block_mask",
            return_value=object(),
        ) as create_mask:
            _, dense_mask = model_module._prepare_document_masks(
                document_ids,
                include_dense_mask=True,
                padding_only=True,
            )

        self.assertEqual(dense_mask.shape, (1, 1, 1, 4))
        torch.testing.assert_close(
            dense_mask,
            torch.tensor([[[[True, True, False, False]]]]),
        )
        mask_mod = create_mask.call_args.args[0]
        self.assertTrue(bool(mask_mod(0, 0, 3, 0)))
        self.assertFalse(bool(mask_mod(0, 0, 3, 3)))

    def test_mixed_flex_schedule_keeps_dense_and_block_masks(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_spaces=["real", "real"],
            attention_backends=["torch", "flex"],
        )
        model = NeoBERT(config)
        layers = nn.ModuleList([RecordingLayer(), RecordingLayer()])
        model.transformer_encoder = layers
        input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        pad_mask = input_ids.ne(0)
        block_mask = object()

        with mock.patch.object(
            model_module,
            "_prepare_document_masks",
            return_value=(block_mask, None),
        ) as prepare_masks:
            output = model(input_ids, pad_mask=pad_mask)

        self.assertEqual(output.shape, (2, 4, 8))
        flex_document_ids = prepare_masks.call_args.args[0]
        self.assertTrue(prepare_masks.call_args.kwargs["padding_only"])
        self.assertFalse(prepare_masks.call_args.kwargs["include_dense_mask"])
        expected_document_ids = torch.where(
            pad_mask,
            torch.zeros_like(pad_mask, dtype=torch.int32),
            torch.full_like(pad_mask, -1, dtype=torch.int32),
        )
        torch.testing.assert_close(flex_document_ids, expected_document_ids)
        for layer in layers:
            dense_mask, key_padding_mask, actual_block_mask, _ = layer.arguments
            self.assertEqual(dense_mask.shape, (2, 1, 1, 4))
            torch.testing.assert_close(key_padding_mask, pad_mask.logical_not())
            self.assertIs(actual_block_mask, block_mask)

    def test_only_flex_layers_receive_block_mask(self):
        block_mask = object()
        for backend, expected in (("torch", None), ("flex", block_mask)):
            config = NeoBERTConfig(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                hidden_act="gelu",
                vocab_size=32,
                max_length=8,
                rope=False,
                attention_spaces=["complex"],
                attention_backends=[backend],
            )
            layer = model_module.EncoderBlock(config, 0)
            attention = RecordingAttention()
            layer.complex_attention = attention
            x = torch.randn(2, 4, 8)
            layer._att_block(
                x,
                pad_mask=None,
                freqs_cis=None,
                key_padding_mask=None,
                block_mask=block_mask,
            )
            self.assertIs(attention.block_mask, expected)

    def test_adapter_does_not_silently_drop_flash_masks(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_spaces=["complex"],
            attention_backends=["flash"],
        )
        attention = model_module.NeoBERTComplexAttention(
            config,
            attention_space="complex",
            attention_backend="flash",
        )
        captured = {}

        def fake_attention(query, key, value, **kwargs):
            captured.update(kwargs)
            return tuple(torch.zeros_like(component) for component in value), None

        attention._complex_attention = fake_attention
        mask = torch.ones(4, 4, dtype=torch.bool)
        attention(
            torch.randn(2, 4, 8),
            attn_mask=mask,
            key_padding_mask=None,
            freqs_cis=None,
        )
        self.assertIs(captured["attn_mask"], mask)

    def test_complex_projection_initialization_matches_real_energy(self):
        initialization_range = 0.02
        config = NeoBERTConfig(
            hidden_size=12,
            num_hidden_layers=3,
            num_attention_heads=3,
            intermediate_size=24,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            decoder_init_range=initialization_range,
            attention_spaces=["complex", "split", "dual"],
            attention_backends=["torch", "torch", "torch"],
        )
        model = NeoBERT(config)

        complex_attention = model.transformer_encoder[0].complex_attention
        split_attention = model.transformer_encoder[1].complex_attention
        dual_attention = model.transformer_encoder[2].complex_attention
        expected_pair_readout = torch.zeros_like(complex_attention.readout)
        expected_pair_readout[0].fill_(1.0)
        expected_dual_readout = torch.zeros_like(dual_attention.readout)
        expected_dual_readout[0].fill_(1.0)
        expected_dual_readout[2].fill_(1.0)
        torch.testing.assert_close(
            complex_attention.readout,
            expected_pair_readout,
        )
        torch.testing.assert_close(
            split_attention.readout,
            expected_pair_readout,
        )
        torch.testing.assert_close(
            dual_attention.readout,
            expected_dual_readout,
        )
        pair_bound = initialization_range / math.sqrt(2.0)
        dual_bound = initialization_range / 2.0
        pair_weights = (
            complex_attention.qkv.weight_real,
            complex_attention.qkv.weight_imag,
            complex_attention.out_proj.weight_real,
            complex_attention.out_proj.weight_imag,
            split_attention.qkv.weight_real,
            split_attention.qkv.weight_split,
            split_attention.out_proj.weight_real,
            split_attention.out_proj.weight_split,
        )
        for weight in pair_weights:
            self.assertLessEqual(weight.detach().abs().max().item(), pair_bound)
        for projection in (dual_attention.qkv, dual_attention.out_proj):
            for component in (projection.primal, projection.dual):
                self.assertLessEqual(
                    component.weight_real.detach().abs().max().item(),
                    dual_bound,
                )
                self.assertLessEqual(
                    component.weight_imag.detach().abs().max().item(),
                    dual_bound,
                )


if __name__ == "__main__":
    unittest.main()
