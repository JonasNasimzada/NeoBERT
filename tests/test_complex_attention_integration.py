import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

import neobert.model.model as model_module
from neobert.model import NeoBERT, NeoBERTConfig, NeoBERTLMHead
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
        prepared_key_padding_mask,
    ):
        self.arguments = (
            pad_mask,
            key_padding_mask,
            block_mask,
            prepared_key_padding_mask,
        )
        return x


class RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mask = None
        self.key_padding_mask = None
        self.block_mask = None
        self.prepared_key_padding_mask = None

    def forward(
        self,
        x,
        attn_mask,
        key_padding_mask,
        freqs_cis,
        block_mask,
        prepared_key_padding_mask=None,
    ):
        self.attn_mask = attn_mask
        self.key_padding_mask = key_padding_mask
        self.block_mask = block_mask
        self.prepared_key_padding_mask = prepared_key_padding_mask
        return torch.zeros_like(x)


class TestComplexAttentionIntegration(unittest.TestCase):
    def test_config_enforces_backend_matrix(self):
        backend_matrix = {
            "real": ("auto", "torch", "flash", "flex"),
            "complex": ("auto", "native", "torch", "flash", "flex"),
            "split": ("auto", "native", "torch", "flash", "flex"),
            "dual": ("auto", "native", "torch", "flash", "flash_fused", "flex"),
            "multispace": ("flash", "flex"),
        }
        all_backends = {"auto", "native", "torch", "flash", "flash_fused", "flex"}
        space_names = {
            "real": "real",
            "complex": "ordinary complex",
            "split": "split-complex",
            "dual": "dual-number",
            "multispace": "multispace",
        }

        for space, allowed in backend_matrix.items():
            with self.subTest(space=space, accepted=True):
                config = NeoBERTConfig(
                    hidden_size=12,
                    num_hidden_layers=len(allowed),
                    num_attention_heads=3,
                    intermediate_size=24,
                    hidden_act="gelu",
                    vocab_size=32,
                    max_length=8,
                    rope=False,
                    attention_spaces=[space] * len(allowed),
                    attention_backends=list(allowed),
                )
                self.assertEqual(config.attention_backends, list(allowed))

            for backend in sorted(all_backends.difference(allowed)):
                with self.subTest(space=space, backend=backend), self.assertRaisesRegex(
                    ValueError,
                    rf"{space_names[space]} layers support only",
                ):
                    NeoBERTConfig(
                        hidden_size=12,
                        num_hidden_layers=1,
                        num_attention_heads=3,
                        intermediate_size=24,
                        hidden_act="gelu",
                        vocab_size=32,
                        max_length=8,
                        rope=False,
                        attention_spaces=[space],
                        attention_backends=[backend],
                    )

    def test_removed_attention_backend_identifiers_are_rejected(self):
        for backend in ("reference", "xformers"):
            with self.subTest(backend=backend, scalar=True), self.assertRaisesRegex(
                ValueError,
                "attention_backend must be 'auto', 'native', 'torch', 'flash', "
                "'flash_fused', or 'flex'",
            ):
                NeoBERTConfig(attention_backend=backend)
            with self.subTest(backend=backend, schedule=True), self.assertRaisesRegex(
                ValueError,
                "attention_backends contains an unknown backend",
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
                    attention_spaces=["complex"],
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

        with self.assertRaisesRegex(
            ValueError,
            "multispace FlashAttention.*no larger than 128",
        ):
            NeoBERTConfig(
                hidden_size=387,
                num_hidden_layers=1,
                num_attention_heads=3,
                intermediate_size=774,
                rope=False,
                attention_spaces=["multispace"],
                attention_backends=["flash"],
            )

        multispace_flex = NeoBERTConfig(
            hidden_size=387,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=774,
            rope=False,
            attention_spaces=["multispace"],
            attention_backends=["flex"],
        )
        self.assertEqual(multispace_flex.dim_head, 129)
        self.assertEqual(multispace_flex.attention_backends, ["flex"])

        with self.assertRaisesRegex(
            ValueError,
            "multispace attention requires num_attention_heads to be divisible by 3",
        ):
            NeoBERTConfig(
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=32,
                rope=False,
                attention_spaces=["multispace"],
                attention_backends=["flash"],
            )

    def test_config_validates_attention_dropout(self):
        self.assertEqual(NeoBERTConfig().attention_dropout, 0.0)
        self.assertEqual(NeoBERTConfig(attention_dropout=0.25).attention_dropout, 0.25)
        for value in (-0.1, 1.0, float("nan"), float("inf"), "invalid"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "attention_dropout"
            ):
                NeoBERTConfig(attention_dropout=value)
        for space in ("real", "complex", "split"):
            with self.subTest(space=space), self.assertRaisesRegex(
                ValueError, "FlexAttention.*attention_dropout"
            ):
                NeoBERTConfig(
                    num_hidden_layers=1,
                    attention_dropout=0.1,
                    attention_spaces=[space],
                    attention_backends=["flex"],
                )

        with self.assertRaisesRegex(
            ValueError, "split-complex FlashAttention.*attention_dropout"
        ):
            NeoBERTConfig(
                num_hidden_layers=1,
                attention_dropout=0.1,
                attention_spaces=["split"],
                attention_backends=["flash"],
            )

        for backend in ("flash", "flash_fused"):
            with self.subTest(backend=backend), self.assertRaisesRegex(
                ValueError, "dual-number FlashAttention.*attention_dropout"
            ):
                NeoBERTConfig(
                    num_hidden_layers=1,
                    attention_dropout=0.1,
                    attention_spaces=["dual"],
                    attention_backends=[backend],
                )

        for backend in ("flash", "flex"):
            with self.subTest(backend=backend), self.assertRaisesRegex(
                ValueError,
                "multispace FlashAttention and FlexAttention.*attention_dropout",
            ):
                NeoBERTConfig(
                    num_hidden_layers=1,
                    attention_dropout=0.1,
                    attention_spaces=["multispace"],
                    attention_backends=[backend],
                )

        dual_flex = NeoBERTConfig(
            num_hidden_layers=1,
            attention_dropout=0.1,
            attention_spaces=["dual"],
            attention_backends=["flex"],
        )
        self.assertEqual(dual_flex.attention_dropout, 0.1)

    def test_complex_adapters_apply_attention_dropout_only_while_training(self):
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
                    attention_dropout=0.25,
                    attention_spaces=[space],
                    attention_backends=["torch"],
                )
                attention = model_module.NeoBERTComplexAttention(
                    config,
                    attention_space=space,
                    attention_backend="torch",
                )
                captured = []

                def fake_attention(query, key, value, *args, **kwargs):
                    captured.append(kwargs["dropout_p"])
                    return (query if args else value), None

                setattr(attention, f"_{space}_attention", fake_attention)
                inputs = torch.randn(2, 4, 8)
                attention.train()
                attention(inputs, None, None, None)
                attention.eval()
                attention(inputs, None, None, None)
                self.assertEqual(captured, [0.25, 0.0])

    def test_real_blocks_apply_attention_dropout_only_while_training(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_dropout=0.25,
            attention_spaces=["real"],
            attention_backends=["torch"],
        )
        inputs = torch.randn(2, 4, 8)

        for block_type in (model_module.EncoderBlock, model_module.NormEncoderBlock):
            with self.subTest(block_type=block_type.__name__):
                block = block_type(config, 0)
                captured = []

                def fake_attention(query, key, value, *args, **kwargs):
                    captured.append(kwargs["dropout_p"])
                    return torch.zeros_like(query)

                with mock.patch.object(
                    model_module, "_real_attention", side_effect=fake_attention
                ):
                    block.train()
                    block._att_block(inputs, None, None)
                    block.eval()
                    block._att_block(inputs, None, None)

                self.assertEqual(captured, [0.25, 0.0])

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

    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile is unavailable")
    def test_floating_padding_mask_compiles_without_graph_breaks(self):
        additive_mask = torch.tensor(
            [[0.0, 0.0, float("-inf"), float("-inf")]],
            dtype=torch.bfloat16,
        )

        def prepare(mask):
            return model_module._prepare_attention_masks(
                mask,
                num_heads=2,
                seq_len=4,
            )

        compiled = torch.compile(prepare, backend="eager", fullgraph=True)
        actual_bias, actual_padding = compiled(additive_mask)
        expected_bias, expected_padding = prepare(additive_mask)
        torch.testing.assert_close(actual_bias, expected_bias)
        torch.testing.assert_close(actual_padding, expected_padding)

    def test_legacy_flash_flag_maps_to_flash_backend(self):
        flash_config = NeoBERTConfig(flash_attention=True)
        torch_config = NeoBERTConfig(flash_attention=False)
        modern_config = NeoBERTConfig(
            flash_attention=True,
            attention_backend="torch",
        )
        self.assertTrue(all(value == "flash" for value in flash_config.attention_backends))
        self.assertTrue(all(value == "torch" for value in torch_config.attention_backends))
        self.assertTrue(all(value == "torch" for value in modern_config.attention_backends))

    def test_ordinary_complex_native_backend_is_accepted(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=8,
            rope=False,
            attention_space="complex",
            attention_backend="native",
        )
        self.assertEqual(config.attention_spaces, ["complex"])
        self.assertEqual(config.attention_backends, ["native"])

    def test_supported_homogeneous_models_take_optimizer_steps(self):
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        labels = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        combinations = (
            ("complex", "native"),
            ("complex", "torch"),
            ("split", "native"),
            ("split", "torch"),
            ("dual", "native"),
            ("dual", "torch"),
        )

        for use_autocast in (False, True):
            for space, backend in combinations:
                with self.subTest(
                    space=space,
                    backend=backend,
                    autocast=use_autocast,
                ):
                    torch.manual_seed(1234)
                    config = NeoBERTConfig(
                        hidden_size=8,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        intermediate_size=16,
                        hidden_act="gelu",
                        vocab_size=19,
                        max_length=4,
                        rope=False,
                        dropout=0.0,
                        attention_space=space,
                        attention_backend=backend,
                    )
                    model = NeoBERTLMHead(config).train()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
                    attention = model.model.transformer_encoder[0].complex_attention
                    before = {
                        name: parameter.detach().clone()
                        for name, parameter in attention.named_parameters()
                    }

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cpu",
                        dtype=torch.bfloat16,
                        enabled=use_autocast,
                    ):
                        logits = model(input_ids)["logits"]
                        loss = F.cross_entropy(
                            logits.float().reshape(-1, config.vocab_size),
                            labels.reshape(-1),
                        )
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()

                    for name, parameter in attention.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(parameter.grad.count_nonzero().item(), 0, name)

                    optimizer.step()
                    self.assertTrue(
                        any(
                            not torch.equal(before[name], parameter.detach())
                            for name, parameter in attention.named_parameters()
                        )
                    )

    def test_nonzero_attention_dropout_models_take_optimizer_steps(self):
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        labels = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        combinations = (
            ("real", "torch"),
            ("complex", "torch"),
            ("complex", "native"),
            ("split", "torch"),
            ("split", "native"),
            ("dual", "torch"),
            ("dual", "native"),
            ("dual", "flex"),
        )

        for index, (space, backend) in enumerate(combinations):
            with self.subTest(space=space, backend=backend):
                torch.manual_seed(8000 + index)
                config = NeoBERTConfig(
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    hidden_act="gelu",
                    vocab_size=19,
                    max_length=4,
                    rope=False,
                    dropout=0.0,
                    attention_dropout=0.25,
                    attention_space=space,
                    attention_backend=backend,
                )
                model = NeoBERTLMHead(config).train()
                block = model.model.transformer_encoder[0]
                attention_parameters = {
                    name: parameter
                    for name, parameter in block.named_parameters()
                    if name.startswith(("qkv.", "wo.", "complex_attention."))
                }
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in attention_parameters.items()
                }
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

                logits = model(input_ids)["logits"]
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size), labels.reshape(-1)
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in attention_parameters.items():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

                optimizer.step()
                self.assertTrue(
                    any(
                        not torch.equal(before[name], parameter.detach())
                        for name, parameter in attention_parameters.items()
                    )
                )

    def test_multispace_attention_partitions_one_packed_qkv_and_concatenates(self):
        hidden_size = 12
        config = NeoBERTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=24,
            hidden_act="gelu",
            vocab_size=19,
            max_length=4,
            rope=False,
            dropout=0.0,
            attention_dropout=0.0,
            attention_space="multispace",
            attention_backend="flash",
        )
        attention = model_module.NeoBERTMultiSpaceAttention(
            config,
            attention_backend="flash",
        ).eval()

        self.assertEqual(attention.space_names, ("complex", "split", "dual"))
        self.assertEqual(attention.num_heads, 3)
        self.assertEqual(attention.heads_per_space, 1)
        self.assertEqual(attention.head_dim, 4)
        self.assertEqual(attention.group_width, 4)
        self.assertEqual(
            tuple(attention.qkv.weight.shape),
            (6 * hidden_size, hidden_size),
        )
        self.assertEqual(
            tuple(attention.out_proj.weight.shape),
            (hidden_size, 2 * hidden_size),
        )
        self.assertFalse(hasattr(attention, "fusion_logits"))
        self.assertFalse(hasattr(attention, "readout"))
        self.assertEqual(
            set(dict(attention.named_parameters())),
            {"qkv.weight", "out_proj.weight"},
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in attention.parameters()),
            8 * hidden_size**2,
        )

        identity = torch.eye(hidden_size)
        with torch.no_grad():
            attention.out_proj.weight.copy_(
                torch.cat((identity, 10.0 * identity), dim=1)
            )

        captures = {}

        def fake_attention(name, first_value, second_value):
            def run(query, key, value, **kwargs):
                captures[name] = (query, key, value, kwargs)
                return (
                    torch.full_like(value[0], first_value),
                    torch.full_like(value[1], second_value),
                ), None

            return mock.Mock(side_effect=run)

        complex_attention = fake_attention("complex", 1.0, 2.0)
        split_attention = fake_attention("split", 3.0, 5.0)
        dual_attention = fake_attention("dual", 7.0, 11.0)
        inputs = torch.randn(2, 3, hidden_size)
        group_width = attention.group_width
        packed_qkv = torch.cat(
            tuple(
                torch.cat(
                    tuple(
                        torch.full(
                            (2, 3, group_width),
                            float(10 * space_index + chunk_index + 1),
                        )
                        for chunk_index in range(6)
                    ),
                    dim=-1,
                )
                for space_index in range(3)
            ),
            dim=-1,
        )
        with (
            mock.patch.object(attention, "_complex_attention", complex_attention),
            mock.patch.object(attention, "_split_attention", split_attention),
            mock.patch.object(attention, "_dual_attention", dual_attention),
            mock.patch.object(
                attention.qkv,
                "forward",
                return_value=packed_qkv,
            ) as qkv_forward,
        ):
            actual = attention(inputs, None, None, None)

        qkv_forward.assert_called_once_with(inputs)
        complex_attention.assert_called_once()
        split_attention.assert_called_once()
        dual_attention.assert_called_once()
        for space_index, branch_name in enumerate(attention.space_names):
            expected_component_values = (
                (10 * space_index + 1, 10 * space_index + 4),
                (10 * space_index + 2, 10 * space_index + 5),
                (10 * space_index + 3, 10 * space_index + 6),
            )
            for argument_index, pair in enumerate(captures[branch_name][:3]):
                for component_index, component in enumerate(pair):
                    self.assertEqual(tuple(component.shape), (2, 1, 3, 4))
                    torch.testing.assert_close(
                        component,
                        torch.full_like(
                            component,
                            float(
                                expected_component_values[argument_index][
                                    component_index
                                ]
                            ),
                        ),
                    )
            kwargs = captures[branch_name][3]
            self.assertEqual(kwargs["backend"], "flash")
            self.assertEqual(kwargs["dropout_p"], 0.0)
            self.assertEqual(kwargs["scale"], config.dim_head**-0.5)
            self.assertIsNone(kwargs["attn_mask"])
            self.assertIsNone(kwargs["key_padding_mask"])
            self.assertIsNone(kwargs["block_mask"])
            self.assertIsNone(kwargs["prepared_key_padding_mask"])

        expected = torch.cat(
            (
                torch.full((2, 3, group_width), 51.0),
                torch.full((2, 3, group_width), 72.0),
                torch.full((2, 3, group_width), 113.0),
            ),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected)

    def test_multispace_flex_routes_one_document_block_mask_to_all_spaces(self):
        config = NeoBERTConfig(
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=24,
            hidden_act="gelu",
            vocab_size=19,
            pad_token_id=0,
            max_length=4,
            rope=False,
            dropout=0.0,
            attention_dropout=0.0,
            attention_space="multispace",
            attention_backend="flex",
        )
        model = NeoBERT(config).eval()
        attention = model.transformer_encoder[0].complex_attention
        block_mask = object()
        document_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        captures = {}

        def fake_attention(space):
            def run(query, key, value, **kwargs):
                captures[space] = kwargs
                return tuple(torch.zeros_like(component) for component in value), None

            return mock.Mock(side_effect=run)

        branch_mocks = {
            "complex": fake_attention("complex"),
            "split": fake_attention("split"),
            "dual": fake_attention("dual"),
        }
        with (
            mock.patch.object(
                attention,
                "_complex_attention",
                branch_mocks["complex"],
            ),
            mock.patch.object(
                attention,
                "_split_attention",
                branch_mocks["split"],
            ),
            mock.patch.object(
                attention,
                "_dual_attention",
                branch_mocks["dual"],
            ),
            mock.patch.object(
                model_module,
                "_prepare_document_masks",
                return_value=block_mask,
            ) as prepare_document_masks,
        ):
            output = model(input_ids, document_ids=document_ids)

        self.assertEqual(output.shape, (1, 4, config.hidden_size))
        prepare_document_masks.assert_called_once_with(
            document_ids,
            padding_only=False,
        )
        self.assertEqual(set(captures), set(attention.space_names))
        for space in attention.space_names:
            branch_mocks[space].assert_called_once()
            kwargs = captures[space]
            self.assertEqual(kwargs["backend"], "flex")
            self.assertEqual(kwargs["dropout_p"], 0.0)
            self.assertIs(kwargs["block_mask"], block_mask)
            self.assertIsNone(kwargs["attn_mask"])
            self.assertIsNone(kwargs["key_padding_mask"])
            self.assertIsNone(kwargs["prepared_key_padding_mask"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_complex_flash_backends_take_low_precision_optimizer_steps(self):
        flash_available = getattr(
            torch.backends.cuda,
            "is_flash_attention_available",
            lambda: True,
        )
        input_ids = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            device="cuda",
        )
        labels = torch.tensor(
            [[2, 3, 4, 5], [6, 7, 8, 9]],
            device="cuda",
        )
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        cases = (
            [(space, "flash") for space in ("complex", "split", "dual")]
            if flash_available()
            else []
        )
        if (
            importlib.util.find_spec("triton") is not None
            and torch.cuda.get_device_capability()[0] >= 8
        ):
            cases.append(("dual", "flash_fused"))
        if not cases:
            self.skipTest("no Flash attention backend is available")
        for index, (space, backend) in enumerate(cases):
            with self.subTest(space=space, backend=backend):
                torch.manual_seed(1234 + index)
                config = NeoBERTConfig(
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    hidden_act="gelu",
                    vocab_size=19,
                    max_length=4,
                    rope=False,
                    dropout=0.0,
                    attention_space=space,
                    attention_backend=backend,
                )
                model = NeoBERTLMHead(config).cuda().train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
                attention = model.model.transformer_encoder[0].complex_attention
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in attention.named_parameters()
                }

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=autocast_dtype):
                    logits = model(input_ids)["logits"]
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, config.vocab_size),
                        labels.reshape(-1),
                    )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in attention.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.count_nonzero().item(), 0, name)

                optimizer.step()
                self.assertTrue(
                    any(
                        not torch.equal(before[name], parameter.detach())
                        for name, parameter in attention.named_parameters()
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_multispace_flash_model_takes_low_precision_optimizer_step(self):
        flash_available = getattr(
            torch.backends.cuda,
            "is_flash_attention_available",
            lambda: True,
        )
        if not flash_available():
            self.skipTest("PyTorch Flash SDPA is unavailable")
        if importlib.util.find_spec("triton") is None:
            self.skipTest("the multispace Flash model requires Triton for dual attention")
        if "A100" not in torch.cuda.get_device_name().upper():
            self.skipTest("the multispace Flash integration test requires an A100")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 0))
        self.assertTrue(torch.cuda.is_bf16_supported())

        spaces = ["multispace"] * 3
        backends = ["flash"] * len(spaces)
        torch.manual_seed(4321)
        config = NeoBERTConfig(
            hidden_size=24,
            num_hidden_layers=len(spaces),
            num_attention_heads=3,
            intermediate_size=48,
            hidden_act="gelu",
            vocab_size=19,
            max_length=4,
            rope=False,
            dropout=0.0,
            attention_dropout=0.0,
            attention_space="multispace",
            attention_backend="flash",
            attention_spaces=spaces,
            attention_backends=backends,
            tie_word_embeddings=True,
            lm_head_bias=False,
        )
        model = NeoBERTLMHead(config).cuda().train()
        blocks = list(model.model.transformer_encoder)
        self.assertEqual([block.attention_space for block in blocks], spaces)
        self.assertEqual([block.attention_backend for block in blocks], backends)
        for block in blocks:
            attention = block.complex_attention
            self.assertEqual(attention.space_names, ("complex", "split", "dual"))
            self.assertEqual(attention.heads_per_space, 1)
            self.assertEqual(attention.group_width, 8)
            self.assertEqual(
                tuple(attention.qkv.weight.shape),
                (6 * config.hidden_size, config.hidden_size),
            )
            self.assertEqual(
                tuple(attention.out_proj.weight.shape),
                (config.hidden_size, 2 * config.hidden_size),
            )
            self.assertFalse(hasattr(attention, "fusion_logits"))
            self.assertFalse(hasattr(attention, "readout"))
            self.assertEqual(
                sum(parameter.numel() for parameter in attention.parameters()),
                8 * config.hidden_size**2,
            )

        before = [
            {
                name: parameter.detach().clone()
                for name, parameter in block.complex_attention.named_parameters()
            }
            for block in blocks
        ]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        input_ids = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            device="cuda",
        )
        labels = torch.tensor(
            [[2, 3, 4, 5], [6, 7, 8, 9]],
            device="cuda",
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)["logits"]
            loss = F.cross_entropy(
                logits.float().reshape(-1, config.vocab_size),
                labels.reshape(-1),
            )
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for layer_index, block in enumerate(blocks):
            for name, parameter in block.complex_attention.named_parameters():
                qualified_name = f"layer {layer_index} {name}"
                self.assertIsNotNone(parameter.grad, qualified_name)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    qualified_name,
                )
                self.assertGreater(
                    parameter.grad.count_nonzero().item(),
                    0,
                    qualified_name,
                )

        optimizer.step()
        for layer_index, block in enumerate(blocks):
            self.assertTrue(
                any(
                    not torch.equal(
                        before[layer_index][name],
                        parameter.detach(),
                    )
                    for name, parameter in block.complex_attention.named_parameters()
                ),
                f"layer {layer_index} attention parameters did not update",
            )

        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected_logits = model(input_ids)["logits"].detach().clone()

        with tempfile.TemporaryDirectory() as temporary_directory:
            model.cpu().save_pretrained(
                temporary_directory,
                safe_serialization=True,
            )
            self.assertTrue(
                (Path(temporary_directory) / "model.safetensors").is_file()
            )
            loaded = (
                NeoBERTLMHead.from_pretrained(temporary_directory).cuda().eval()
            )
            self.assertEqual(loaded.config.attention_spaces, spaces)
            self.assertEqual(loaded.config.attention_backends, backends)
            self.assertIs(
                loaded.decoder.weight,
                loaded.model.encoder.weight,
            )
            for layer_index, loaded_block in enumerate(
                loaded.model.transformer_encoder
            ):
                loaded_parameters = dict(
                    loaded_block.complex_attention.named_parameters()
                )
                expected_parameters = dict(
                    blocks[layer_index].complex_attention.named_parameters()
                )
                self.assertEqual(
                    set(loaded_parameters),
                    set(expected_parameters),
                )
                for name, parameter in loaded_parameters.items():
                    torch.testing.assert_close(
                        parameter.cpu(),
                        expected_parameters[name],
                        rtol=0.0,
                        atol=0.0,
                    )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                actual_logits = loaded(input_ids)["logits"]
            torch.testing.assert_close(
                actual_logits,
                expected_logits,
                rtol=0.0,
                atol=0.0,
            )
        torch.cuda.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_multispace_flash_model_runs_production_geometry(self):
        flash_available = getattr(
            torch.backends.cuda,
            "is_flash_attention_available",
            lambda: True,
        )
        if not flash_available():
            self.skipTest("PyTorch Flash SDPA is unavailable")
        if importlib.util.find_spec("triton") is None:
            self.skipTest("the multispace Flash model requires Triton for dual attention")
        if "A100" not in torch.cuda.get_device_name().upper():
            self.skipTest("the multispace Flash production test requires an A100")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 0))
        self.assertTrue(torch.cuda.is_bf16_supported())

        config_path = (
            Path(__file__).resolve().parents[1]
            / "conf"
            / "model"
            / "attention-ablation-multispace.yaml"
        )
        values = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        self.assertIsInstance(values, dict)
        values.update(vocab_size=30_522, max_length=1024, pad_token_id=0)
        config = NeoBERTConfig(**values)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_hidden_layers, 9)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.dim_head, 64)
        self.assertEqual(config.intermediate_size, 2464)
        self.assertTrue(config.rope)
        self.assertEqual(
            config.attention_spaces,
            ["multispace"] * 9,
        )
        self.assertEqual(config.attention_backends, ["flash"] * 9)

        torch.manual_seed(8765)
        model = NeoBERTLMHead(config).cuda().train()
        for layer_index, block in enumerate(model.model.transformer_encoder):
            attention = block.complex_attention
            self.assertEqual(attention.space_names, ("complex", "split", "dual"))
            self.assertEqual(attention.num_heads, 12)
            self.assertEqual(attention.heads_per_space, 4)
            self.assertEqual(attention.head_dim, 64)
            self.assertEqual(attention.group_width, 256)
            self.assertEqual(
                tuple(attention.qkv.weight.shape),
                (4608, 768),
            )
            self.assertEqual(
                tuple(attention.out_proj.weight.shape),
                (768, 1536),
            )
            self.assertFalse(hasattr(attention, "fusion_logits"))
            self.assertFalse(hasattr(attention, "readout"))
            self.assertEqual(
                sum(parameter.numel() for parameter in attention.parameters()),
                4_718_592,
                f"layer {layer_index} attention parameter count",
            )
            layer_parameters = sum(
                parameter.numel() * (2 if parameter.is_complex() else 1)
                for parameter in block.parameters()
            )
            self.assertEqual(layer_parameters, 8_504_832)
        trainable_parameters = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(trainable_parameters, 99_985_152)
        self.assertIs(model.decoder.weight, model.model.encoder.weight)
        input_ids = torch.randint(
            1,
            config.vocab_size,
            (1, config.max_length),
            device="cuda",
        )
        labels = torch.randint(
            0,
            config.vocab_size,
            input_ids.shape,
            device="cuda",
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)["logits"]
            loss = F.cross_entropy(
                logits.float().reshape(-1, config.vocab_size),
                labels.reshape(-1),
            )
        self.assertEqual(logits.shape, (1, 1024, config.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for layer_index, block in enumerate(model.model.transformer_encoder):
            gradients = [
                parameter.grad
                for parameter in block.complex_attention.parameters()
            ]
            self.assertTrue(
                all(gradient is not None for gradient in gradients),
                f"layer {layer_index} has a missing attention gradient",
            )
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients),
                f"layer {layer_index} has a non-finite attention gradient",
            )
            self.assertTrue(
                any(gradient.count_nonzero().item() > 0 for gradient in gradients),
                f"layer {layer_index} has only zero attention gradients",
            )
        torch.cuda.synchronize()

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

    def test_flex_adapter_forwards_mask_routing_to_complex_attention(self):
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

        def fake_attention(query, key, value, *args, **kwargs):
            captured.update(kwargs)
            return query, None

        attention._complex_attention = fake_attention
        block_mask = object()
        attn_mask = torch.ones(4, 4, dtype=torch.bool)
        key_padding_mask = torch.zeros(1, 4, dtype=torch.bool)
        prepared_key_padding_mask = object()
        attention(
            torch.randn(1, 4, 8),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            freqs_cis=None,
            block_mask=block_mask,
            prepared_key_padding_mask=prepared_key_padding_mask,
        )

        self.assertIs(captured["attn_mask"], attn_mask)
        self.assertIs(captured["key_padding_mask"], key_padding_mask)
        self.assertIs(captured["prepared_key_padding_mask"], prepared_key_padding_mask)
        self.assertIs(captured["block_mask"], block_mask)

    def test_padding_metadata_is_prepared_once_for_flash_backends(self):
        key_padding_mask = torch.tensor([[False, False, True, True]])
        sentinel = object()
        schedules = (
            (["real"], ["flash"]),
            (["complex"], ["flash"]),
            (["split"], ["flash"]),
            (["dual"], ["flash"]),
            (["dual"], ["flash_fused"]),
            (["multispace"], ["flash"]),
            (
                ["real", "complex", "split", "dual"],
                ["flash", "flash", "flash", "flash"],
            ),
        )
        for spaces, backends in schedules:
            with self.subTest(spaces=spaces), mock.patch(
                "complex_attention.prepare_key_padding_mask",
                return_value=sentinel,
            ) as prepare:
                actual = model_module._prepare_backend_padding_metadata(
                    key_padding_mask,
                    spaces,
                    backends,
                )

            self.assertIs(actual, sentinel)
            prepare.assert_called_once_with(key_padding_mask)

    def test_split_and_dual_flex_do_not_prepare_flash_padding_metadata(self):
        key_padding_mask = torch.tensor([[False, False, True, True]])
        with mock.patch("complex_attention.prepare_key_padding_mask") as prepare:
            actual = model_module._prepare_backend_padding_metadata(
                key_padding_mask,
                ["split", "dual"],
                ["flex", "flex"],
            )

        self.assertIsNone(actual)
        prepare.assert_not_called()

    def test_model_forwards_never_prepare_backend_padding_metadata(self):
        input_ids = torch.tensor([[1, 2, 0, 0]])
        pad_mask = input_ids.ne(0)
        for ngpt, model_type in (
            (False, NeoBERT),
            (True, model_module.NormNeoBERT),
        ):
            with self.subTest(ngpt=ngpt):
                config = NeoBERTConfig(
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    hidden_act="gelu",
                    vocab_size=32,
                    max_length=8,
                    rope=False,
                    ngpt=ngpt,
                    attention_spaces=["real"],
                    attention_backends=["flash"],
                )
                model = model_type(config)
                layer = RecordingLayer()
                model.transformer_encoder = nn.ModuleList([layer])

                with mock.patch.object(
                    model_module,
                    "_prepare_backend_padding_metadata",
                    side_effect=AssertionError("padding metadata was prepared"),
                ) as prepare:
                    output = model(input_ids, pad_mask=pad_mask)

                self.assertEqual(output.shape, (1, 4, 8))
                prepare.assert_not_called()
                self.assertIsNone(layer.arguments[3])

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

    def test_dual_adapter_applies_rope_and_forwards_flash_padding_metadata(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=4,
            rope=True,
            attention_spaces=["dual"],
            attention_backends=["flash"],
        )
        attention = model_module.NeoBERTComplexAttention(
            config,
            attention_space="dual",
            attention_backend="flash",
        )
        primal_qkv = torch.randn(1, 4, 24)
        dual_qkv = torch.randn_like(primal_qkv)
        frequencies = precompute_freqs_cis(config.dim_head, 4)
        prepared_padding = object()
        captured = {}

        def fake_attention(query, key, value, **kwargs):
            captured["query"] = query
            captured["key"] = key
            captured.update(kwargs)
            return value, None

        attention._dual_attention = fake_attention
        with mock.patch.object(
            attention.qkv,
            "forward_real",
            return_value=(primal_qkv, dual_qkv),
        ):
            output = attention(
                torch.randn(1, 4, 8),
                attn_mask=None,
                key_padding_mask=torch.zeros(1, 4, dtype=torch.bool),
                freqs_cis=frequencies,
                prepared_key_padding_mask=prepared_padding,
            )

        self.assertEqual(output.shape, (1, 4, 8))
        expected = []
        for qkv in (primal_qkv, dual_qkv):
            query, key, _ = qkv.view(1, 4, 2, 12).chunk(3, dim=-1)
            expected.append(apply_rotary_emb(query, key, frequencies))
        for component in range(2):
            torch.testing.assert_close(
                captured["query"][component],
                expected[component][0].transpose(1, 2),
            )
            torch.testing.assert_close(
                captured["key"][component],
                expected[component][1].transpose(1, 2),
            )
        self.assertEqual(captured["backend"], "flash")
        self.assertIs(captured["prepared_key_padding_mask"], prepared_padding)
        self.assertIsNone(captured["block_mask"])

    def test_dual_adapter_matches_explicit_product_rule_and_readout(self):
        torch.manual_seed(2026)
        config = NeoBERTConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=8,
            hidden_act="gelu",
            vocab_size=16,
            max_length=3,
            rope=False,
            attention_spaces=["dual"],
            attention_backends=["torch"],
        )
        attention = model_module.NeoBERTComplexAttention(
            config,
            attention_space="dual",
            attention_backend="torch",
        ).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64)

        qkv_primal, qkv_dual = attention.qkv.linear(x).chunk(2, dim=-1)
        q0, k0, v0 = (
            component.transpose(1, 2)
            for component in qkv_primal.view(2, 3, 1, 12).chunk(3, dim=-1)
        )
        q1, k1, v1 = (
            component.transpose(1, 2)
            for component in qkv_dual.view(2, 3, 1, 12).chunk(3, dim=-1)
        )
        scale = config.dim_head**-0.5
        score0 = torch.matmul(q0, k0.transpose(-2, -1)) * scale
        score1 = (
            torch.matmul(q1, k0.transpose(-2, -1))
            + torch.matmul(q0, k1.transpose(-2, -1))
        ) * scale
        probability0 = torch.softmax(score0, dim=-1)
        probability1 = probability0 * (
            score1 - (probability0 * score1).sum(dim=-1, keepdim=True)
        )
        output0 = torch.matmul(probability0, v0)
        output1 = torch.matmul(probability0, v1) + torch.matmul(probability1, v0)
        output0 = output0.transpose(1, 2).reshape(2, 3, 4)
        output1 = output1.transpose(1, 2).reshape(2, 3, 4)
        weight_primal, weight_dual = attention.out_proj.linear.weight.chunk(
            2, dim=0
        )
        projected0 = F.linear(output0, weight_primal)
        projected1 = F.linear(output1, weight_primal) + F.linear(
            output0, weight_dual
        )
        expected = (
            attention.readout[0] * projected0
            + attention.readout[1] * projected1
        )

        actual = attention(x, None, None, None)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_dual_flex_adapter_drops_dense_padding_inputs(self):
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_act="gelu",
            vocab_size=32,
            max_length=4,
            rope=False,
            attention_spaces=["dual"],
            attention_backends=["flex"],
        )
        attention = model_module.NeoBERTComplexAttention(
            config,
            attention_space="dual",
            attention_backend="flex",
        )
        captured = {}

        def fake_attention(query, key, value, **kwargs):
            captured.update(kwargs)
            return value, None

        attention._dual_attention = fake_attention
        block_mask = object()
        attention(
            torch.randn(1, 4, 8),
            attn_mask=torch.eye(4, dtype=torch.bool),
            key_padding_mask=torch.zeros(1, 4, dtype=torch.bool),
            freqs_cis=None,
            block_mask=block_mask,
            prepared_key_padding_mask=object(),
        )

        self.assertIsNone(captured["attn_mask"])
        self.assertIsNone(captured["key_padding_mask"])
        self.assertIsNone(captured["prepared_key_padding_mask"])
        self.assertIs(captured["block_mask"], block_mask)

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
            block_mask = model_module._prepare_document_masks(
                document_ids,
                padding_only=True,
            )

        mask_mod = create_mask.call_args.args[0]
        self.assertIsNotNone(block_mask)
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
            attention_backends=["torch", "flex"],
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
            return_value=block_mask,
        ) as prepare_masks:
            output = model(input_ids, document_ids=document_ids)

        self.assertEqual(output.shape, (1, 4, 8))
        self.assertIs(prepare_masks.call_args.args[0], document_ids)
        expected_dense_mask = model_module._prepare_dense_document_mask(document_ids)
        for layer in layers:
            dense_mask, key_padding_mask, actual_block_mask, prepared = layer.arguments
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
            return_value=block_mask,
        ) as prepare_masks:
            output = model(input_ids, pad_mask=pad_mask)

        self.assertEqual(output.shape, (2, 4, 8))
        flex_document_ids = prepare_masks.call_args.args[0]
        self.assertTrue(prepare_masks.call_args.kwargs["padding_only"])
        expected_document_ids = torch.where(
            pad_mask,
            torch.zeros_like(pad_mask, dtype=torch.int32),
            torch.full_like(pad_mask, -1, dtype=torch.int32),
        )
        torch.testing.assert_close(flex_document_ids, expected_document_ids)
        for layer in layers:
            dense_mask, key_padding_mask, actual_block_mask, prepared = layer.arguments
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

        def fake_attention(query, key, value, *args, **kwargs):
            captured.update(kwargs)
            return tuple(torch.zeros_like(component) for component in query), None

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

    def test_pair_projection_initialization_matches_real_energy(self):
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
        expected_dual_readout = torch.ones_like(dual_attention.readout)
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
        pair_weights = (
            complex_attention.qkv.weight.real,
            complex_attention.qkv.weight.imag,
            complex_attention.out_proj.weight.real,
            complex_attention.out_proj.weight.imag,
            *split_attention.qkv.linear.weight.chunk(2, dim=0),
            *split_attention.out_proj.linear.weight.chunk(2, dim=0),
        )
        for weight in pair_weights:
            self.assertLessEqual(weight.detach().abs().max().item(), pair_bound)
        for projection in (dual_attention.qkv, dual_attention.out_proj):
            for weight in projection.linear.weight.chunk(2, dim=0):
                self.assertLessEqual(weight.detach().abs().max().item(), pair_bound)


if __name__ == "__main__":
    unittest.main()
