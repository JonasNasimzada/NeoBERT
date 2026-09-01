import math
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .rotary import apply_rotary_emb


def _reshape_qkv(x: Tensor, num_heads: int, head_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, seq_len, _ = x.shape
    return x.view(batch_size, seq_len, num_heads, head_dim * 3).chunk(3, dim=-1)


def _to_attention_layout(x: Tensor) -> Tensor:
    return x.transpose(1, 2)


def _from_attention_layout(x: Tensor) -> Tensor:
    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


def _apply_pair_rope(
    query: tuple[Tensor, Tensor],
    key: tuple[Tensor, Tensor],
    freqs_cis: Tensor,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    query_first, key_first = apply_rotary_emb(query[0], key[0], freqs_cis)
    query_second, key_second = apply_rotary_emb(query[1], key[1], freqs_cis)
    return (query_first, query_second), (key_first, key_second)


class MultiSpaceStreamPool:
    """Two persistent side streams shared by all multispace layers."""

    def __init__(self) -> None:
        self.streams = ()

    def set_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            self.streams = ()
        elif not self.streams or self.streams[0].device != device:
            self.streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))


class NeoBERTComplexAttention(nn.Module):
    def __init__(self, config, attention_space: str, attention_backend: str) -> None:
        super().__init__()
        try:
            from complex_attention import (
                DualLinear,
                SplitComplexLinear,
                complex_attention,
                dual_attention,
                split_complex_attention,
            )
        except (ImportError, OSError) as error:
            raise ImportError(
                "Install ComplexAttention with `pip install -e /Users/joni/PycharmProjects/ComplexAttention --no-deps`"
            ) from error

        self.space = attention_space
        self.backend = attention_backend
        self.num_heads = config.num_attention_heads
        self.head_dim = config.dim_head
        self.rope = config.rope
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self._complex_attention = complex_attention
        self._split_attention = split_complex_attention
        self._dual_attention = dual_attention

        if self.space == "complex":
            kwargs = {"bias": False, "dtype": torch.cfloat}
            self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, **kwargs)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, **kwargs)
            readout = torch.zeros(2, config.hidden_size)
            readout[0].fill_(1.0)
        elif self.space == "split":
            self.qkv = SplitComplexLinear(config.hidden_size, config.hidden_size * 3, bias=False)
            self.out_proj = SplitComplexLinear(config.hidden_size, config.hidden_size, bias=False)
            readout = torch.zeros(2, config.hidden_size)
            readout[0].fill_(1.0)
        elif self.space == "dual":
            self.qkv = DualLinear(config.hidden_size, config.hidden_size * 3, bias=False)
            self.out_proj = DualLinear(config.hidden_size, config.hidden_size, bias=False)
            readout = torch.ones(2, config.hidden_size)
        else:
            raise ValueError(f"unsupported complex attention space: {self.space}")
        self.readout = nn.Parameter(readout)

    def reset_parameters(self, initialization_range: float) -> None:
        if self.space == "complex":
            bound = initialization_range / math.sqrt(2.0)
            with torch.no_grad():
                for layer in (self.qkv, self.out_proj):
                    layer.weight.real.uniform_(-bound, bound)
                    layer.weight.imag.uniform_(-bound, bound)
        elif self.space == "split":
            bound = initialization_range / math.sqrt(2.0)
            layers = (self.qkv, self.out_proj)
            with torch.no_grad():
                for layer in layers:
                    layer.linear.weight.uniform_(-bound, bound)
        elif self.space == "dual":
            bound = initialization_range / math.sqrt(2.0)
            layers = (self.qkv, self.out_proj)
            with torch.no_grad():
                for layer in layers:
                    layer.linear.weight.uniform_(-bound, bound)
        with torch.no_grad():
            self.readout.zero_()
            self.readout[0].fill_(1.0)
            if self.space == "dual":
                self.readout[1].fill_(1.0)

    def _complex_forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        block_mask: Any,
        prepared_key_padding_mask: Any,
    ) -> Tensor:
        output, _ = self._complex_attention(
            (x, torch.zeros_like(x)),
            self.qkv,
            self.out_proj,
            self.num_heads,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            scale=self.head_dim**-0.5,
            dropout_p=self.attention_dropout if self.training else 0.0,
            backend=self.backend,
            block_mask=block_mask,
            prepared_key_padding_mask=prepared_key_padding_mask,
            rotary_emb=apply_rotary_emb if self.rope else None,
            freqs_cis=freqs_cis,
        )
        return sum(
            component * coefficient.to(component)
            for component, coefficient in zip(output, self.readout)
        )

    def _split_forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        block_mask: Any,
        prepared_key_padding_mask: Any,
    ) -> Tensor:
        qkv_real, qkv_split = self.qkv.forward_real(x)
        q_real, k_real, v_real = _reshape_qkv(qkv_real, self.num_heads, self.head_dim)
        q_split, k_split, v_split = _reshape_qkv(qkv_split, self.num_heads, self.head_dim)
        if self.rope:
            q_real, k_real = apply_rotary_emb(q_real, k_real, freqs_cis)
            q_split, k_split = apply_rotary_emb(q_split, k_split, freqs_cis)

        uses_block_mask = self.backend == "flex" and block_mask is not None
        direct_mask = None if uses_block_mask else attn_mask
        direct_key_padding = None if uses_block_mask else key_padding_mask
        direct_prepared_padding = None if uses_block_mask else prepared_key_padding_mask
        output, _ = self._split_attention(
            (_to_attention_layout(q_real), _to_attention_layout(q_split)),
            (_to_attention_layout(k_real), _to_attention_layout(k_split)),
            (_to_attention_layout(v_real), _to_attention_layout(v_split)),
            attn_mask=direct_mask,
            key_padding_mask=direct_key_padding,
            scale=self.head_dim**-0.5,
            dropout_p=self.attention_dropout if self.training else 0.0,
            backend=self.backend,
            block_mask=block_mask,
            prepared_key_padding_mask=direct_prepared_padding,
        )
        output = (_from_attention_layout(output[0]), _from_attention_layout(output[1]))
        return self.out_proj.forward_readout(output, self.readout)

    def _dual_forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        block_mask: Any,
        prepared_key_padding_mask: Any,
    ) -> Tensor:
        qkv_primal, qkv_dual = self.qkv.forward_real(x)
        q_primal, k_primal, v_primal = _reshape_qkv(
            qkv_primal, self.num_heads, self.head_dim
        )
        q_dual, k_dual, v_dual = _reshape_qkv(
            qkv_dual, self.num_heads, self.head_dim
        )
        query = q_primal, q_dual
        key = k_primal, k_dual
        value = v_primal, v_dual
        if self.rope:
            query, key = _apply_pair_rope(query, key, freqs_cis)

        uses_block_mask = self.backend == "flex" and block_mask is not None
        direct_mask = None if uses_block_mask else attn_mask
        direct_key_padding = None if uses_block_mask else key_padding_mask
        direct_prepared_padding = None if uses_block_mask else prepared_key_padding_mask
        output, _ = self._dual_attention(
            tuple(_to_attention_layout(component) for component in query),
            tuple(_to_attention_layout(component) for component in key),
            tuple(_to_attention_layout(component) for component in value),
            attn_mask=direct_mask,
            key_padding_mask=direct_key_padding,
            scale=self.head_dim**-0.5,
            dropout_p=self.attention_dropout if self.training else 0.0,
            backend=self.backend,
            block_mask=block_mask if self.backend == "flex" else None,
            prepared_key_padding_mask=direct_prepared_padding,
        )
        output = tuple(_from_attention_layout(component) for component in output)
        return self.out_proj.forward_readout(output, self.readout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        block_mask: Any = None,
        prepared_key_padding_mask: Any = None,
    ) -> Tensor:
        if key_padding_mask is None and prepared_key_padding_mask is not None:
            key_padding_mask = getattr(
                prepared_key_padding_mask,
                "key_padding_mask",
                None,
            )
            if key_padding_mask is None:
                raise ValueError(
                    "prepared_key_padding_mask must come from prepare_key_padding_mask"
                )
        if self.space == "complex":
            return self._complex_forward(
                x,
                attn_mask,
                key_padding_mask,
                freqs_cis,
                block_mask,
                prepared_key_padding_mask,
            )
        if self.space == "split":
            return self._split_forward(
                x,
                attn_mask,
                key_padding_mask,
                freqs_cis,
                block_mask,
                prepared_key_padding_mask,
            )
        return self._dual_forward(
            x,
            attn_mask,
            key_padding_mask,
            freqs_cis,
            block_mask,
            prepared_key_padding_mask,
        )


class NeoBERTMultiSpaceAttention(nn.Module):
    """Real-MHA-style equal head groups using three scalar algebras."""

    space_names = ("complex", "split", "dual")

    def __init__(
        self,
        config,
        attention_backend: str,
        stream_pool: Optional[MultiSpaceStreamPool] = None,
    ) -> None:
        super().__init__()
        if attention_backend not in ("flash", "flex"):
            raise ValueError(
                "multispace attention requires backend='flash' or backend='flex'"
            )
        if config.num_attention_heads % len(self.space_names) != 0:
            raise ValueError(
                "multispace attention requires an equal number of complex, "
                "split-complex, and dual-number heads"
            )
        try:
            from complex_attention import (
                complex_dot_product_attention,
                dual_attention,
                split_complex_attention,
            )
        except (ImportError, OSError) as error:
            raise ImportError(
                "Install ComplexAttention before using multispace attention"
            ) from error

        self.backend = attention_backend
        self.num_heads = config.num_attention_heads
        self.heads_per_space = self.num_heads // len(self.space_names)
        self.head_dim = config.dim_head
        self.group_width = self.heads_per_space * self.head_dim
        self.rope = config.rope
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        if self.group_width * len(self.space_names) != config.hidden_size:
            raise ValueError("multispace head groups must exactly cover hidden_size")

        # Exactly like a packed real MHA projection, except every algebra head
        # has two scalar components. Rows are laid out as three equal space
        # groups, each containing component-0 QKV followed by component-1 QKV.
        self.qkv = nn.Linear(
            config.hidden_size,
            6 * config.hidden_size,
            bias=False,
        )
        # Each algebra head emits two real scalar components. As in ordinary
        # multi-head attention, retain every head channel, concatenate them,
        # and let one shared output projection perform all cross-head mixing.
        self.out_proj = nn.Linear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self._complex_attention = complex_dot_product_attention
        self._split_attention = split_complex_attention
        self._dual_attention = dual_attention
        self._stream_pool = (
            stream_pool if stream_pool is not None else MultiSpaceStreamPool()
        )
        self._use_cuda_streams = getattr(config, "multispace_cuda_streams", True)
        if self._use_cuda_streams:
            self._stream_pool.set_device(self.qkv.weight.device)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        device = self.qkv.weight.device
        self._stream_pool.set_device(
            device if self._use_cuda_streams else torch.device("cpu")
        )
        return result

    def reset_parameters(self, initialization_range: float) -> None:
        pair_bound = initialization_range / math.sqrt(2.0)
        with torch.no_grad():
            self.qkv.weight.uniform_(-pair_bound, pair_bound)
            self.out_proj.weight.uniform_(
                -initialization_range,
                initialization_range,
            )

    def _space_forward(
        self,
        space: str,
        packed_space_qkv: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        block_mask: Any,
        prepared_key_padding_mask: Any,
    ) -> tuple[Tensor, Tensor]:
        qkv_first, qkv_second = packed_space_qkv.chunk(2, dim=-1)
        query_first, key_first, value_first = _reshape_qkv(
            qkv_first,
            self.heads_per_space,
            self.head_dim,
        )
        query_second, key_second, value_second = _reshape_qkv(
            qkv_second,
            self.heads_per_space,
            self.head_dim,
        )
        query = query_first, query_second
        key = key_first, key_second
        value = value_first, value_second
        if self.rope:
            query, key = _apply_pair_rope(query, key, freqs_cis)

        uses_block_mask = self.backend == "flex" and block_mask is not None
        direct_mask = None if uses_block_mask else attn_mask
        direct_key_padding = None if uses_block_mask else key_padding_mask
        direct_prepared_padding = None if uses_block_mask else prepared_key_padding_mask
        attention_function = {
            "complex": self._complex_attention,
            "split": self._split_attention,
            "dual": self._dual_attention,
        }[space]
        output, _ = attention_function(
            tuple(_to_attention_layout(component) for component in query),
            tuple(_to_attention_layout(component) for component in key),
            tuple(_to_attention_layout(component) for component in value),
            attn_mask=direct_mask,
            key_padding_mask=direct_key_padding,
            scale=self.head_dim**-0.5,
            dropout_p=self.attention_dropout if self.training else 0.0,
            backend=self.backend,
            block_mask=block_mask if self.backend == "flex" else None,
            prepared_key_padding_mask=direct_prepared_padding,
        )
        return tuple(_from_attention_layout(component) for component in output)

    # AOTAutograd cannot yet lower the cross-stream Flex backward as one graph.
    # Keep the fork/join eager so compiled training uses eager stream autograd.
    @torch.compiler.disable(recursive=False)
    def _parallel_space_forwards(
        self,
        packed_space_qkv: tuple[Tensor, ...],
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        block_mask: Any,
        prepared_key_padding_mask: Any,
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        def run(space_index: int) -> tuple[Tensor, Tensor]:
            return self._space_forward(
                self.space_names[space_index],
                packed_space_qkv[space_index],
                attn_mask,
                key_padding_mask,
                freqs_cis,
                block_mask,
                prepared_key_padding_mask,
            )

        current_stream = torch.cuda.current_stream(packed_space_qkv[0].device)
        split_stream, dual_stream = self._stream_pool.streams
        split_stream.wait_stream(current_stream)
        dual_stream.wait_stream(current_stream)

        with torch.cuda.stream(dual_stream):
            packed_space_qkv[2].record_stream(dual_stream)
            dual_output = run(2)
        with torch.cuda.stream(split_stream):
            packed_space_qkv[1].record_stream(split_stream)
            split_output = run(1)
        complex_output = run(0)

        current_stream.wait_stream(split_stream)
        current_stream.wait_stream(dual_stream)
        for output in (split_output, dual_output):
            for component in output:
                component.record_stream(current_stream)
        return complex_output, split_output, dual_output

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        block_mask: Any = None,
        prepared_key_padding_mask: Any = None,
    ) -> Tensor:
        if key_padding_mask is None and prepared_key_padding_mask is not None:
            key_padding_mask = getattr(
                prepared_key_padding_mask,
                "key_padding_mask",
                None,
            )
            if key_padding_mask is None:
                raise ValueError(
                    "prepared_key_padding_mask must come from prepare_key_padding_mask"
                )
        packed_space_qkv = self.qkv(x).chunk(len(self.space_names), dim=-1)
        streams = self._stream_pool.streams
        if streams and streams[0].device != x.device:
            raise RuntimeError(
                "multispace CUDA streams require one model device per process"
            )
        group_outputs = (
            self._parallel_space_forwards(
                packed_space_qkv,
                attn_mask,
                key_padding_mask,
                freqs_cis,
                block_mask,
                prepared_key_padding_mask,
            )
            if streams
            else tuple(
                self._space_forward(
                    space,
                    packed_space_qkv[index],
                    attn_mask,
                    key_padding_mask,
                    freqs_cis,
                    block_mask,
                    prepared_key_padding_mask,
                )
                for index, space in enumerate(self.space_names)
            )
        )

        concatenated_heads = torch.cat(
            tuple(
                component
                for group_output in group_outputs
                for component in group_output
            ),
            dim=-1,
        )
        return self.out_proj(concatenated_heads)
