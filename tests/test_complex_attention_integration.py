import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

import neobert.model.model as model_module
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.model.rotary import apply_rotary_emb, precompute_freqs_cis


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
        dual_tangent_mask_mod,
        prepared_key_padding_mask,
    ):
        self.arguments = (
            pad_mask,
            key_padding_mask,
            block_mask,
            dual_tangent_mask_mod,
            prepared_key_padding_mask,
        )
        return x


class RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mask = None
        self.key_padding_mask = None
        self.block_mask = None
        self.tangent_mask_mod = None
        self.prepared_key_padding_mask = None

    def forward(
        self,
        x,
        attn_mask,
        key_padding_mask,
        freqs_cis,
        block_mask,
        tangent_mask_mod=None,
        prepared_key_padding_mask=None,
    ):
        self.attn_mask = attn_mask
        self.key_padding_mask = key_padding_mask
        self.block_mask = block_mask
        self.tangent_mask_mod = tangent_mask_mod
        self.prepared_key_padding_mask = prepared_key_padding_mask
        return torch.zeros_like(x)


class TestComplexAttentionIntegration(unittest.TestCase):
    def test_shipped_mixed_config_uses_exact_split_backend(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "conf"
            / "model"
            / "optibertneo-mixed-198m.yaml"
        )
        lines = config_path.read_text().splitlines()

        def yaml_list(name):
            start = lines.index(f"{name}:") + 1
            values = []
            for line in lines[start:]:
                if line and not line[0].isspace():
                    break
                stripped = line.strip()
                if stripped.startswith("- "):
                    values.append(stripped[2:])
            return values

        attention_spaces = yaml_list("attention_spaces")
        attention_backends = yaml_list("attention_backends")
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=len(attention_spaces),
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_spaces=attention_spaces,
            attention_backends=attention_backends,
        )

        split_backends = [
            backend
            for space, backend in zip(
                config.attention_spaces,
                config.attention_backends,
            )
            if space == "split"
        ]
        self.assertEqual(split_backends, ["torch"] * 9)

    def test_split_layers_reject_narrow_fused_backends(self):
        for backend in ("reference", "xformers", "flash", "flex"):
            with self.subTest(backend=backend), self.assertRaisesRegex(
                ValueError,
                "split-complex layers support only auto, native, or torch",
            ):
                NeoBERTConfig(
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    hidden_act="gelu",
                    vocab_size=32,
                    max_length=8,
                    rope=False,
                    attention_spaces=["split"],
                    attention_backends=[backend],
                )

    def test_config_rejects_nonpositive_attention_geometry(self):
        for kwargs in (
            {"hidden_size": 0, "num_attention_heads": 1},
            {"hidden_size": 8, "num_attention_heads": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    NeoBERTConfig(**kwargs)

    def test_all_zero_floating_padding_mask_is_binary_and_fully_padded(self):
        bias, key_padding_mask = model_module._prepare_attention_masks(
            torch.zeros(2, 4, dtype=torch.float32),
            num_heads=2,
            seq_len=4,
        )
        self.assertTrue(key_padding_mask.all())
        self.assertTrue(torch.isneginf(bias).all())

    def test_integer_padding_masks_must_be_binary(self):
        bias, key_padding_mask = model_module._prepare_attention_masks(
            torch.tensor([[1, 0, 1, 0]], dtype=torch.int64),
            num_heads=2,
            seq_len=4,
        )
        torch.testing.assert_close(
            key_padding_mask,
            torch.tensor([[False, True, False, True]]),
        )
        self.assertTrue(torch.isneginf(bias[..., 1::2]).all())

        for pad_mask in (
            torch.tensor([[0, 2, 1, 0]], dtype=torch.int64),
            torch.tensor([[0, -1, 1, 0]], dtype=torch.int32),
        ):
            with self.subTest(pad_mask=pad_mask), self.assertRaisesRegex(
                ValueError,
                "binary 0/1",
            ):
                model_module._prepare_attention_masks(
                    pad_mask,
                    num_heads=2,
                    seq_len=4,
                )

    def test_complex_padding_masks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "real floating-point dtype"):
            model_module._prepare_attention_masks(
                torch.tensor([[1 + 0j, 0 + 0j]]),
                num_heads=1,
                seq_len=2,
            )

    def test_legacy_flash_flag_maps_to_historical_backend(self):
        xformers_config = NeoBERTConfig(flash_attention=True)
        torch_config = NeoBERTConfig(flash_attention=False)
        modern_config = NeoBERTConfig(
            flash_attention=True,
            attention_backend="torch",
        )
        self.assertTrue(all(value == "xformers" for value in xformers_config.attention_backends))
        self.assertTrue(all(value == "torch" for value in torch_config.attention_backends))
        self.assertTrue(all(value == "torch" for value in modern_config.attention_backends))

    def test_meta_state_assignment_regenerates_nonpersistent_rope(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=True,
        )
        source = NeoBERT(config)
        with torch.device("meta"):
            target = NeoBERT(config)
        target.load_state_dict(source.state_dict(), assign=True)
        self.assertEqual(target.freqs_cis.device.type, "meta")

        output = target(torch.tensor([[1, 2, 3, 0]]))

        self.assertEqual(output.device.type, "cpu")
        self.assertEqual(target.freqs_cis.device.type, "cpu")
        torch.testing.assert_close(target.freqs_cis, source.freqs_cis)

    def test_half_model_runs_all_complex_attention_spaces(self):
        input_ids = torch.tensor([[1, 2, 3, 0]])
        for space in ("complex", "split", "dual"):
            with self.subTest(space=space):
                config = NeoBERTConfig(
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    hidden_act="gelu",
                    vocab_size=32,
                    max_length=8,
                    rope=False,
                    attention_spaces=[space],
                    attention_backends=["torch"],
                )
                model = NeoBERT(config).half()
                output = model(input_ids)
                self.assertEqual(output.dtype, torch.float16)
                self.assertEqual(output.shape, (1, 4, 8))
                self.assertTrue(torch.isfinite(output).all())

    def test_rope_accepts_noncontiguous_last_dimension(self):
        base_query = torch.randn(2, 5, 3, 16)
        base_key = torch.randn_like(base_query)
        query = base_query[..., ::2]
        key = base_key[..., ::2]
        self.assertFalse(query.is_contiguous())
        frequencies = precompute_freqs_cis(8, 5)

        actual = apply_rotary_emb(query, key, frequencies)
        expected = apply_rotary_emb(
            query.contiguous(),
            key.contiguous(),
            frequencies,
        )

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_model_dtype_conversion_preserves_complex_rope_master(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=True,
        )
        model = NeoBERT(config)
        expected = precompute_freqs_cis(4, 8, dtype=torch.float64)

        model.to(dtype=torch.float16)
        self.assertEqual(model.freqs_cis.dtype, torch.complex128)
        self.assertGreater(model.freqs_cis.imag.abs().max().item(), 0.0)
        torch.testing.assert_close(model.freqs_cis.cpu(), expected)

        model.double()
        self.assertEqual(model.freqs_cis.dtype, torch.complex128)
        torch.testing.assert_close(model.freqs_cis.cpu(), expected)

    def test_model_rejects_backend_dependent_mask_broadcasting(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
        )
        model = NeoBERT(config)
        input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        with self.assertRaisesRegex(ValueError, "same shape"):
            model(input_ids, pad_mask=torch.ones(1, 4, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            model(input_ids, document_ids=torch.zeros_like(input_ids, dtype=torch.float32))

    def test_flex_adapter_drops_prepared_dense_padding_metadata(self):
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
            attention_backends=["flex"],
        )
        attention = model_module.NeoBERTComplexAttention(
            config,
            attention_space="complex",
            attention_backend="flex",
        )
        captured = {}

        def fake_attention(query, key, value, **kwargs):
            captured.update(kwargs)
            return value, None

        attention._complex_attention = fake_attention
        block_mask = object()
        attention(
            torch.randn(1, 4, 8),
            attn_mask=torch.ones(4, 4, dtype=torch.bool),
            key_padding_mask=torch.zeros(1, 4, dtype=torch.bool),
            freqs_cis=None,
            block_mask=block_mask,
            prepared_key_padding_mask=object(),
        )

        self.assertIsNone(captured["attn_mask"])
        self.assertIsNone(captured["key_padding_mask"])
        self.assertIsNone(captured["prepared_key_padding_mask"])
        self.assertIs(captured["block_mask"], block_mask)

    def test_padding_metadata_is_prepared_once_for_reusable_backends(self):
        key_padding_mask = torch.tensor([[False, False, True, True]])
        sentinel = object()
        with mock.patch(
            "complex_attention.prepare_key_padding_mask",
            return_value=sentinel,
        ) as prepare:
            actual = model_module._prepare_backend_padding_metadata(
                key_padding_mask,
                ["complex", "split", "dual"],
                ["flash", "xformers", "flash"],
            )

        self.assertIs(actual, sentinel)
        prepare.assert_called_once_with(key_padding_mask)

    def test_dual_streaming_attention_does_not_prepare_fused_padding_metadata(self):
        key_padding_mask = torch.tensor([[False, False, True, True]])
        with mock.patch("complex_attention.prepare_key_padding_mask") as prepare:
            actual = model_module._prepare_backend_padding_metadata(
                key_padding_mask,
                ["dual"],
                ["flash"],
            )

        self.assertIsNone(actual)
        prepare.assert_not_called()

    def test_rope_preserves_float64_precision(self):
        query = torch.randn(2, 5, 3, 8, dtype=torch.float64)
        key = torch.randn_like(query)
        frequencies = precompute_freqs_cis(8, 5, dtype=torch.float64)

        actual_query, actual_key = apply_rotary_emb(query, key, frequencies)
        complex_query = torch.view_as_complex(query.reshape(2, 5, 3, 4, 2))
        complex_key = torch.view_as_complex(key.reshape(2, 5, 3, 4, 2))
        broadcast_frequencies = frequencies.view(1, 5, 1, 4)
        expected_query = torch.view_as_real(
            complex_query * broadcast_frequencies
        ).flatten(-2)
        expected_key = torch.view_as_real(
            complex_key * broadcast_frequencies
        ).flatten(-2)

        self.assertEqual(actual_query.dtype, torch.float64)
        self.assertEqual(actual_key.dtype, torch.float64)
        torch.testing.assert_close(actual_query, expected_query, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(actual_key, expected_key, rtol=1e-13, atol=1e-13)

    def test_rope_rejects_odd_head_dimension_at_construction(self):
        with self.assertRaisesRegex(ValueError, "even attention head dimension"):
            NeoBERTConfig(
                hidden_size=10,
                num_attention_heads=2,
                rope=True,
            )

    def test_rope_frequencies_are_registered_nonpersistent_buffer(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=True,
        )
        model = NeoBERT(config)

        self.assertIn("freqs_cis", dict(model.named_buffers()))
        self.assertNotIn("freqs_cis", model.state_dict())

    def test_dual_adapter_keeps_autocast_projection_dtype(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_spaces=["dual"],
            attention_backends=["torch"],
        )
        attention = model_module.NeoBERTComplexAttention(
            config,
            attention_space="dual",
            attention_backend="torch",
        )
        captured = {}

        def fake_attention(query, key, value, **kwargs):
            captured["query_dtype"] = query[0][0].dtype
            captured["compute_dtype"] = kwargs.get("compute_dtype")
            return value, None

        attention._dual_attention = fake_attention
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = attention(
                torch.randn(2, 4, 8),
                attn_mask=None,
                key_padding_mask=None,
                freqs_cis=None,
            )

        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(captured["query_dtype"], torch.bfloat16)
        self.assertIsNone(captured["compute_dtype"])

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
            _, tangent_mask_mod = model_module._prepare_document_masks(
                document_ids,
                include_tangent_mask_mod=True,
                padding_only=True,
            )

        mask_mod = create_mask.call_args.args[0]
        self.assertIs(tangent_mask_mod, mask_mod)
        self.assertTrue(bool(mask_mod(0, 0, 3, 0)))
        self.assertFalse(bool(mask_mod(0, 0, 3, 3)))

    def test_dense_document_mask_is_shared_and_excludes_padded_queries(self):
        document_ids = torch.tensor([[0, 0, 1, -1]], dtype=torch.int32)
        actual = model_module._prepare_dense_document_mask(document_ids)
        expected = torch.tensor(
            [
                [
                    [
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, True, False],
                        [False, False, False, False],
                    ]
                ]
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_mixed_packed_schedule_keeps_dense_and_block_document_masks(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_spaces=["complex", "split"],
            attention_backends=["flex", "torch"],
        )
        model = NeoBERT(config)
        layers = nn.ModuleList([RecordingLayer(), RecordingLayer()])
        model.transformer_encoder = layers
        input_ids = torch.tensor([[1, 2, 3, 0]])
        document_ids = torch.tensor([[0, 0, 1, -1]], dtype=torch.int32)
        block_mask = object()

        with mock.patch.object(
            model_module,
            "_prepare_document_masks",
            return_value=(block_mask, None),
        ) as prepare_masks:
            output = model(input_ids, document_ids=document_ids)

        self.assertEqual(output.shape, (1, 4, 8))
        self.assertIs(prepare_masks.call_args.args[0], document_ids)
        expected_dense_mask = model_module._prepare_dense_document_mask(document_ids)
        for layer in layers:
            dense_mask, key_padding_mask, actual_block_mask, _, prepared = layer.arguments
            torch.testing.assert_close(dense_mask, expected_dense_mask)
            self.assertIsNone(key_padding_mask)
            self.assertIs(actual_block_mask, block_mask)
            self.assertIsNone(prepared)

    def test_packed_document_schedule_rejects_direct_flash(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_spaces=["real"],
            attention_backends=["flash"],
        )
        model = NeoBERT(config)

        with self.assertRaisesRegex(ValueError, "direct FlashAttention"):
            model(
                torch.tensor([[1, 2, 3, 0]]),
                document_ids=torch.tensor([[0, 0, 1, -1]], dtype=torch.int32),
            )

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
        self.assertFalse(prepare_masks.call_args.kwargs["include_tangent_mask_mod"])
        expected_document_ids = torch.where(
            pad_mask,
            torch.zeros_like(pad_mask, dtype=torch.int32),
            torch.full_like(pad_mask, -1, dtype=torch.int32),
        )
        torch.testing.assert_close(flex_document_ids, expected_document_ids)
        for layer in layers:
            dense_mask, key_padding_mask, actual_block_mask, _, prepared = layer.arguments
            self.assertEqual(dense_mask.shape, (2, 1, 1, 4))
            torch.testing.assert_close(key_padding_mask, pad_mask.logical_not())
            self.assertIs(actual_block_mask, block_mask)
            self.assertIsNone(prepared)

    def test_only_flex_layers_receive_block_mask_and_drop_dense_mask(self):
        block_mask = object()
        dense_mask = torch.eye(4, dtype=torch.bool)
        for backend, expected_block, expected_mask in (
            ("torch", None, dense_mask),
            ("flex", block_mask, None),
        ):
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
                pad_mask=dense_mask,
                freqs_cis=None,
                key_padding_mask=None,
                block_mask=block_mask,
            )
            self.assertIs(attention.block_mask, expected_block)
            self.assertIs(attention.attn_mask, expected_mask)

    def test_dual_flex_receives_exact_tangent_mask_callback(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_spaces=["dual"],
            attention_backends=["flex"],
        )
        layer = model_module.EncoderBlock(config, 0)
        attention = RecordingAttention()
        layer.complex_attention = attention
        block_mask = object()

        def tangent_mask_mod(batch, head, query_index, key_index):
            return query_index == key_index

        layer._att_block(
            torch.randn(2, 4, 8),
            pad_mask=None,
            freqs_cis=None,
            key_padding_mask=None,
            block_mask=block_mask,
            dual_tangent_mask_mod=tangent_mask_mod,
        )

        self.assertIs(attention.block_mask, block_mask)
        self.assertIs(attention.tangent_mask_mod, tangent_mask_mod)

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

    def test_real_flash_bridge_preserves_explicit_mask(self):
        query = torch.randn(2, 4, 2, 3)
        mask = torch.ones(2, 1, 1, 4, dtype=torch.bool)
        fused_output = torch.zeros(2, 2, 4, 3)
        config = SimpleNamespace(attention_backend="flash")

        with mock.patch(
            "complex_attention.efficient_attention",
            return_value=fused_output,
        ) as attention:
            output = model_module._real_attention(
                query,
                query,
                query,
                mask,
                None,
                config,
            )

        self.assertEqual(output.shape, query.shape)
        self.assertIs(attention.call_args.kwargs["attn_mask"], mask)

    def test_real_torch_bridge_applies_key_padding_without_bias(self):
        query = torch.randn(2, 4, 2, 3)
        key_padding_mask = torch.tensor(
            [[False, False, True, True], [False, True, True, True]]
        )
        config = SimpleNamespace(attention_backend="torch")

        actual = model_module._real_attention(
            query,
            query,
            query,
            None,
            key_padding_mask,
            config,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            query.transpose(1, 2),
            query.transpose(1, 2),
            attn_mask=key_padding_mask.logical_not()[:, None, None, :],
        ).transpose(1, 2)
        torch.testing.assert_close(actual, expected)

    def test_real_torch_and_auto_bridges_preserve_finite_cancellation(self):
        magnitude = 1.0e30
        query = torch.tensor([[[[magnitude, magnitude]]]], dtype=torch.float32)
        key = torch.tensor(
            [[[[magnitude, -magnitude]], [[0.0, 0.0]]]],
            dtype=torch.float32,
        )
        value = torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.float32)
        expected = torch.full((1, 1, 1, 1), 1.5, dtype=torch.float32)

        for backend in ("torch", "auto"):
            with self.subTest(backend=backend):
                actual = model_module._real_attention(
                    query,
                    key,
                    value,
                    None,
                    None,
                    SimpleNamespace(attention_backend=backend),
                    scale=1.0e-30,
                )
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)

    def test_real_flex_bridge_rejects_unrepresented_mask(self):
        query = torch.randn(2, 4, 2, 3)
        config = SimpleNamespace(attention_backend="flex")

        with self.assertRaisesRegex(ValueError, "requires block_mask"):
            model_module._real_attention(
                query,
                query,
                query,
                torch.ones(2, 1, 1, 4, dtype=torch.bool),
                None,
                config,
            )

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
