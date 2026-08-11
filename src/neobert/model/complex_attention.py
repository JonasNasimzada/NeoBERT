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


class NeoBERTComplexAttention(nn.Module):
    def __init__(self, config, attention_space: str, attention_backend: str) -> None:
        super().__init__()
        try:
            from complex_attention import (
                ComplexLinear,
                DualLinear,
                SplitComplexLinear,
                complex_attention,
                dual_attention,
                split_complex_attention,
            )
        except ImportError as error:
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
            self.qkv = ComplexLinear(config.hidden_size, config.hidden_size * 3, bias=False)
            self.out_proj = ComplexLinear(config.hidden_size, config.hidden_size, bias=False)
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
            layers = (self.qkv, self.out_proj)
            with torch.no_grad():
                for layer in layers:
                    layer.weight_real.uniform_(-bound, bound)
                    layer.weight_imag.uniform_(-bound, bound)
        elif self.space == "split":
            bound = initialization_range / math.sqrt(2.0)
            layers = (self.qkv, self.out_proj)
            with torch.no_grad():
                for layer in layers:
                    layer.weight_real.uniform_(-bound, bound)
                    layer.weight_split.uniform_(-bound, bound)
        elif self.space == "dual":
            bound = initialization_range / math.sqrt(2.0)
            layers = (self.qkv, self.out_proj)
            with torch.no_grad():
                for layer in layers:
                    layer.weight_primal.uniform_(-bound, bound)
                    layer.weight_dual.uniform_(-bound, bound)
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
        qkv_real, qkv_imag = self.qkv.forward_real(x)
        q_real, k_real, v_real = _reshape_qkv(qkv_real, self.num_heads, self.head_dim)
        q_imag, k_imag, v_imag = _reshape_qkv(qkv_imag, self.num_heads, self.head_dim)
        query = q_real, q_imag
        key = k_real, k_imag
        value = v_real, v_imag
        if self.rope:
            query, key = _apply_pair_rope(query, key, freqs_cis)
        uses_block_mask = self.backend == "flex" and block_mask is not None
        direct_mask = None if uses_block_mask else attn_mask
        direct_key_padding = None if uses_block_mask else key_padding_mask
        direct_prepared_padding = None if uses_block_mask else prepared_key_padding_mask
        output, _ = self._complex_attention(
            tuple(_to_attention_layout(component) for component in query),
            tuple(_to_attention_layout(component) for component in key),
            tuple(_to_attention_layout(component) for component in value),
            attn_mask=direct_mask,
            key_padding_mask=direct_key_padding,
            scale=self.head_dim**-0.5,
            dropout_p=self.attention_dropout if self.training else 0.0,
            backend=self.backend,
            block_mask=block_mask,
            prepared_key_padding_mask=direct_prepared_padding,
        )
        output = tuple(_from_attention_layout(component) for component in output)
        return self.out_proj.forward_readout(output, self.readout)

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
