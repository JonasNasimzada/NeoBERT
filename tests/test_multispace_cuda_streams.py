import importlib.util
import unittest
from unittest import mock

import torch

from neobert.model import NeoBERT, NeoBERTConfig
from neobert.model.complex_attention import NeoBERTMultiSpaceAttention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestMultiSpaceCUDAStreams(unittest.TestCase):
    @staticmethod
    def config(
        *, layers: int = 1, cuda_streams: bool = True
    ) -> NeoBERTConfig:
        return NeoBERTConfig(
            hidden_size=192,
            num_hidden_layers=layers,
            num_attention_heads=3,
            intermediate_size=384,
            hidden_act="gelu",
            vocab_size=32,
            max_length=32,
            rope=False,
            dropout=0.0,
            attention_dropout=0.0,
            attention_space="multispace",
            attention_backend="flash",
            multispace_cuda_streams=cuda_streams,
        )

    def test_model_shares_two_persistent_side_streams(self):
        with torch.device("cuda"):
            direct_attention = NeoBERTMultiSpaceAttention(
                self.config(), attention_backend="flash"
            )
        self.assertEqual(len(direct_attention._stream_pool.streams), 2)

        model = NeoBERT(self.config(layers=3)).cuda()
        attentions = [
            block.complex_attention for block in model.transformer_encoder
        ]
        pool = attentions[0]._stream_pool

        self.assertTrue(
            all(attention._stream_pool is pool for attention in attentions)
        )
        self.assertEqual(len(pool.streams), 2)
        self.assertIsNot(pool.streams[0], pool.streams[1])
        self.assertNotIn(torch.cuda.current_stream(), pool.streams)
        self.assertFalse(any("stream" in name for name in model.state_dict()))

        streams = pool.streams
        model.half()
        self.assertIs(pool.streams, streams)
        model.cpu()
        self.assertEqual(pool.streams, ())

    def test_each_algebra_uses_its_own_execution_stream(self):
        attention = NeoBERTMultiSpaceAttention(
            self.config(), attention_backend="flash"
        ).cuda()
        current_stream = torch.cuda.current_stream()
        split_stream, dual_stream = attention._stream_pool.streams
        observed = {}

        def record_space(space, packed_qkv, *_args):
            observed[space] = torch.cuda.current_stream().cuda_stream
            width = attention.group_width
            return (
                packed_qkv[..., :width].contiguous(),
                packed_qkv[..., width : 2 * width].contiguous(),
            )

        inputs = torch.randn(
            2,
            8,
            attention.num_heads * attention.head_dim,
            device="cuda",
        )
        with mock.patch.object(attention, "_space_forward", side_effect=record_space):
            output = attention(inputs, None, None, None)
        torch.cuda.synchronize()

        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(
            observed,
            {
                "complex": current_stream.cuda_stream,
                "split": split_stream.cuda_stream,
                "dual": dual_stream.cuda_stream,
            },
        )

    def test_parallel_forward_and_backward_match_serial(self):
        flash_available = getattr(
            torch.backends.cuda,
            "is_flash_attention_available",
            lambda: True,
        )
        if not flash_available() or importlib.util.find_spec("triton") is None:
            self.skipTest("multispace FlashAttention is unavailable")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 is required")

        torch.manual_seed(1234)
        serial = NeoBERTMultiSpaceAttention(
            self.config(cuda_streams=False), attention_backend="flash"
        ).cuda()
        parallel = NeoBERTMultiSpaceAttention(
            self.config(), attention_backend="flash"
        ).cuda()
        parallel.load_state_dict(serial.state_dict())
        self.assertEqual(serial._stream_pool.streams, ())

        values = torch.randn(2, 16, 192, device="cuda")
        serial_input = values.clone().requires_grad_()
        parallel_input = values.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            serial_output = serial(serial_input, None, None, None)
            parallel_output = parallel(parallel_input, None, None, None)
        serial_output.float().square().mean().backward()
        parallel_output.float().square().mean().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            parallel_output,
            serial_output,
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            parallel_input.grad,
            serial_input.grad,
            rtol=5e-3,
            atol=1e-5,
        )
        parallel_parameters = dict(parallel.named_parameters())
        self.assertEqual(
            set(parallel_parameters),
            set(dict(serial.named_parameters())),
        )
        for name, expected in serial.named_parameters():
            actual = parallel_parameters[name]
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                rtol=5e-3,
                atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
