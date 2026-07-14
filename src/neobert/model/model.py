# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import math
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import scaled_dot_product_attention

from typing import Any, Dict, List, Optional
from functools import partial

try:
    from xformers.ops import SwiGLU
except (ImportError, OSError):
    SwiGLU = None

from datasets import Dataset

from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput

from tqdm import tqdm

from .complex_attention import NeoBERTComplexAttention
from .rmsnorm import RMSNorm
from .rotary import precompute_freqs_cis, apply_rotary_emb


def _valid_tokens_from_padding_mask(pad_mask):
    if pad_mask.dtype == torch.bool or not torch.is_floating_point(pad_mask):
        return pad_mask.bool()
    has_one, has_binary_values, has_additive_values = torch.stack(
        (
            (pad_mask == 1).any(),
            ((pad_mask == 0) | (pad_mask == 1)).all(),
            ((pad_mask == 0) | torch.isneginf(pad_mask)).all(),
        )
    ).tolist()
    if has_one and has_binary_values:
        return pad_mask > 0
    if has_additive_values:
        return pad_mask == 0
    raise ValueError(
        "floating pad_mask must be binary 0/1 or an additive padding mask containing only 0/-inf"
    )


def _prepare_attention_masks(pad_mask, num_heads, seq_len):
    if pad_mask is None:
        return None, None
    if pad_mask.dim() != 2:
        raise ValueError("pad_mask must have shape (batch, sequence)")
    if pad_mask.size(1) != seq_len:
        raise ValueError("pad_mask sequence length must match input_ids")

    valid_tokens = _valid_tokens_from_padding_mask(pad_mask)
    key_padding_mask = valid_tokens.logical_not()
    mask_dtype = pad_mask.dtype if torch.is_floating_point(pad_mask) else torch.float32
    additive_mask = torch.zeros(
        pad_mask.shape,
        dtype=mask_dtype,
        device=pad_mask.device,
    ).masked_fill(key_padding_mask, float("-inf"))

    attention_bias = additive_mask[:, None, None, :]
    return attention_bias, key_padding_mask


def _prepare_document_masks(document_ids, include_dense_mask, padding_only=False):
    if document_ids is None:
        return None, None
    if document_ids.dim() != 2:
        raise ValueError("document_ids must have shape (batch, sequence)")

    from torch.nn.attention.flex_attention import create_block_mask

    def document_mask(batch, head, query_index, key_index):
        key_document = document_ids[batch, key_index]
        if padding_only:
            return key_document >= 0
        query_document = document_ids[batch, query_index]
        return (query_document == key_document) & (query_document >= 0)

    batch_size, sequence_length = document_ids.shape
    block_mask = create_block_mask(
        document_mask,
        B=batch_size,
        H=1,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=document_ids.device,
    )
    dense_mask = None
    if include_dense_mask:
        if padding_only:
            dense_mask = document_ids[:, None, None, :] >= 0
        else:
            dense_mask = document_ids[:, None, :, None] == document_ids[:, None, None, :]
            dense_mask = dense_mask & (document_ids[:, None, :, None] >= 0)
    return block_mask, dense_mask


def _document_ids_from_key_padding_mask(key_padding_mask):
    return torch.where(
        key_padding_mask.logical_not(),
        torch.zeros_like(key_padding_mask, dtype=torch.int32),
        torch.full_like(key_padding_mask, -1, dtype=torch.int32),
    )


def _real_attention(
    query,
    key,
    value,
    attn_bias,
    key_padding_mask,
    config,
    scale=None,
    backend=None,
    block_mask=None,
):
    backend = config.attention_backend if backend is None else backend
    if backend == "auto":
        backend = "torch"
    if backend == "xformers":
        try:
            from complex_attention import efficient_attention
        except (ImportError, OSError) as error:
            raise ImportError(
                "Install ComplexAttention before using attention_backend='xformers'"
            ) from error
        return efficient_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_bias,
            key_padding_mask=key_padding_mask,
            scale=scale,
            backend="xformers",
        ).transpose(1, 2)
    if backend == "flash":
        try:
            from complex_attention import efficient_attention
        except (ImportError, OSError) as error:
            raise ImportError(
                "Install ComplexAttention before using attention_backend='flash'"
            ) from error
        return efficient_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_bias,
            key_padding_mask=key_padding_mask,
            scale=scale,
            backend="flash",
        ).transpose(1, 2)
    if backend == "flex":
        if attn_bias is not None or key_padding_mask is not None:
            raise ValueError("real FlexAttention requires masking to be represented only by block_mask")
        try:
            from complex_attention import efficient_attention
        except (ImportError, OSError) as error:
            raise ImportError(
                "Install ComplexAttention before using attention_backend='flex'"
            ) from error
        return efficient_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=scale,
            backend="flex",
            block_mask=block_mask,
        ).transpose(1, 2)
    if backend != "torch":
        raise ValueError(f"unsupported real attention backend: {backend}")

    bias = None if attn_bias is None else attn_bias.to(device=query.device)
    if bias is not None and bias.dtype != torch.bool:
        bias = bias.to(dtype=query.dtype)
    if key_padding_mask is not None:
        padding_keep = key_padding_mask.logical_not()[:, None, None, :]
        if bias is None:
            bias = padding_keep
        elif bias.dtype == torch.bool:
            bias = bias & padding_keep
        else:
            bias = bias.masked_fill(padding_keep.logical_not(), float("-inf"))
    return scaled_dot_product_attention(
        query=query.transpose(1, 2),
        key=key.transpose(1, 2),
        value=value.transpose(1, 2),
        attn_mask=bias,
        dropout_p=0.0,
        scale=scale,
    ).transpose(1, 2)


class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        rope: bool = True,
        norm_eps: float = 1e-06,
        hidden_act: str = "SwiGLU",
        vocab_size: int = 32064,
        pad_token_id: int = 0,
        max_length: int = 1024,
        flash_attention: bool = True,
        attention_space: str = "real",
        attention_backend: str = "auto",
        attention_spaces: Optional[List[str]] = None,
        attention_backends: Optional[List[str]] = None,
        dual_tangent_chunk_size: int = 128,
        base_scale: float = 1.0 / (960.0**0.5),
        ngpt: bool = False,
        embedding_rms_norm: bool = False,
        tie_word_embeddings: bool = False,
        lm_head_bias: bool = True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.rope = rope
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.flash_attention = flash_attention
        valid_spaces = ("real", "complex", "split", "dual")
        valid_backends = ("auto", "native", "reference", "torch", "xformers", "flash", "flex")
        if attention_space not in valid_spaces:
            raise ValueError("attention_space must be 'real', 'complex', 'split', or 'dual'")
        if attention_backend not in valid_backends:
            raise ValueError(
                "attention_backend must be 'auto', 'native', 'reference', 'torch', 'xformers', 'flash', or 'flex'"
            )
        if dual_tangent_chunk_size <= 0:
            raise ValueError("dual_tangent_chunk_size must be positive")
        if attention_spaces is None:
            attention_spaces = [attention_space] * num_hidden_layers
        if attention_backends is None:
            attention_backends = [attention_backend] * num_hidden_layers
        if len(attention_spaces) != num_hidden_layers:
            raise ValueError("attention_spaces must contain one value per hidden layer")
        if len(attention_backends) != num_hidden_layers:
            raise ValueError("attention_backends must contain one value per hidden layer")
        if any(space not in valid_spaces for space in attention_spaces):
            raise ValueError("attention_spaces contains an unknown scalar space")
        if any(backend not in valid_backends for backend in attention_backends):
            raise ValueError("attention_backends contains an unknown backend")
        for layer_space, layer_backend in zip(attention_spaces, attention_backends):
            if layer_space == "real" and layer_backend in ("native", "reference"):
                raise ValueError("real layers support auto, torch, xformers, flash, or flex")
            if layer_space == "complex" and layer_backend in ("native", "reference"):
                raise ValueError("ordinary complex layers support auto, torch, xformers, flash, or flex")
            if layer_space == "split" and layer_backend == "reference":
                raise ValueError("split-complex layers do not use attention_backend='reference'")
        self.attention_space = attention_space
        self.attention_backend = attention_backend
        self.attention_spaces = list(attention_spaces)
        self.attention_backends = list(attention_backends)
        self.dual_tangent_chunk_size = dual_tangent_chunk_size
        self.base_scale = base_scale
        self.ngpt = ngpt
        self.embedding_rms_norm = embedding_rms_norm
        self.lm_head_bias = lm_head_bias
        self.kwargs = kwargs


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig, layer_index: int):
        super().__init__()

        self.config = config
        self.layer_index = layer_index
        self.attention_space = config.attention_spaces[layer_index]
        self.attention_backend = config.attention_backends[layer_index]

        # Attention
        if self.attention_space == "real":
            self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
            self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.complex_attention = None
        else:
            self.qkv = None
            self.wo = None
            self.complex_attention = NeoBERTComplexAttention(
                config,
                self.attention_space,
                self.attention_backend,
            )
        self.resid_dropout = nn.Dropout(config.dropout)

        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                if SwiGLU is None:
                    raise RuntimeError("hidden_act='SwiGLU' requires an installed xFormers package")
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
                # avoid RuntimeError due to misaligned operand
                multiple_of = 8
                intermediate_size = int(2 * config.intermediate_size / 3)
                intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)
            case "gelu":
                self.ffn = nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                    nn.GELU(),
                    nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                )

        self.attention_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )
        self.ffn_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        block_mask=None,
        dual_attention_mask: torch.Tensor = None,
    ):
        x = x + self._att_block(
            self.attention_norm(x),
            pad_mask,
            freqs_cis,
            key_padding_mask,
            block_mask,
            dual_attention_mask,
        )
        x = x + self._ff_block(self.ffn_norm(x))
        return x

    def _att_block(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        block_mask=None,
        dual_attention_mask: torch.Tensor = None,
    ):
        layer_block_mask = block_mask if self.attention_backend == "flex" else None
        if self.complex_attention is not None:
            if self.attention_space == "dual" and self.attention_backend == "flex":
                attention_mask = dual_attention_mask
            else:
                attention_mask = None if key_padding_mask is not None else pad_mask
            return self.resid_dropout(
                self.complex_attention(
                    x,
                    attention_mask,
                    key_padding_mask,
                    freqs_cis,
                    layer_block_mask,
                )
            )

        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3).chunk(3, axis=-1)

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        uses_block_mask = self.attention_backend == "flex" and layer_block_mask is not None
        attention_bias = None if key_padding_mask is not None or uses_block_mask else pad_mask
        direct_key_padding = None if uses_block_mask else key_padding_mask
        attn = _real_attention(
            xq,
            xk,
            xv,
            attention_bias,
            direct_key_padding,
            self.config,
            backend=self.attention_backend,
            block_mask=layer_block_mask,
        )

        return self.resid_dropout(self.wo(attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.config.dim_head)))

    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


class NormEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig, layer_index: int):
        super().__init__()

        self.config = config
        self.layer_index = layer_index
        self.attention_space = config.attention_spaces[layer_index]
        self.attention_backend = config.attention_backends[layer_index]
        if self.attention_space != "real":
            raise ValueError("complex attention schedules currently require ngpt=False")

        # Attention
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.c_fc = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.ffn_dropout = nn.Dropout(config.dropout)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale
        self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling * torch.ones(config.hidden_size))

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(config.hidden_size))

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(config.hidden_size))

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(self.suv_init_scaling * torch.ones(2 * config.intermediate_size))

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        block_mask=None,
    ):
        x_attn = self._att_block(x, pad_mask, freqs_cis, key_padding_mask, block_mask)

        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_attn)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        x_ff = self._ff_block(x)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_ff)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        return x

    def _att_block(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        block_mask=None,
    ):
        batch_size, seq_len, _ = x.shape
        layer_block_mask = block_mask if self.attention_backend == "flex" else None

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3).chunk(3, axis=-1)

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, 1, self.config.num_attention_heads, self.config.hidden_size // self.config.num_attention_heads
        )
        xq = sqk * self.justnorm(xq)
        xk = sqk * self.justnorm(xk)

        softmax_scale = (self.config.hidden_size / self.config.num_attention_heads) ** 0.5

        uses_block_mask = self.attention_backend == "flex" and layer_block_mask is not None
        attention_bias = None if key_padding_mask is not None or uses_block_mask else pad_mask
        direct_key_padding = None if uses_block_mask else key_padding_mask
        attn = _real_attention(
            xq,
            xk,
            xv,
            attention_bias,
            direct_key_padding,
            self.config,
            scale=softmax_scale,
            backend=self.attention_backend,
            block_mask=layer_block_mask,
        )

        return self.resid_dropout(self.wo(attn.reshape(batch_size, seq_len, self.config.hidden_size)))

    def _ff_block(self, x: torch.Tensor):
        uv = self.c_fc(x)
        suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.hidden_size**0.5))
        uv = suv * uv

        u, v = torch.chunk(uv, 2, dim=-1)
        x = u * self.silu(v)
        x = self.mlp_c_proj(x)

        return self.ffn_dropout(x)


class NeoBERTPreTrainedModel(PreTrainedModel):
    config_class = NeoBERTConfig
    _supports_cache_class = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)
        elif isinstance(module, NeoBERTComplexAttention):
            module.reset_parameters(self.config.decoder_init_range)


class NeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embedding_norm = (
            RMSNorm(config.hidden_size, config.norm_eps)
            if config.embedding_rms_norm
            else nn.Identity()
        )

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        else:
            self.positional_embedding = nn.Embedding(config.max_length + 1, config.hidden_size, padding_idx=config.pad_token_id)

        self.transformer_encoder = nn.ModuleList()
        for layer_index in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config, layer_index))

        self.layer_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, src, pad_mask=None, document_ids=None):
        uses_flex = any(
            backend == "flex" for backend in self.config.attention_backends
        )
        uses_only_flex = all(
            backend == "flex" for backend in self.config.attention_backends
        )
        if document_ids is not None:
            if document_ids.shape != src.shape:
                raise ValueError("document_ids must have the same shape as input_ids")
            if pad_mask is not None:
                raise ValueError("packed document_ids cannot be combined with pad_mask")
            if not uses_only_flex:
                raise ValueError("packed document masking requires attention_backend='flex'")
        pad_mask, key_padding_mask = _prepare_attention_masks(
            pad_mask,
            self.config.num_attention_heads,
            src.shape[1],
        )
        flex_document_ids = document_ids
        if document_ids is None and key_padding_mask is not None and uses_flex:
            flex_document_ids = _document_ids_from_key_padding_mask(key_padding_mask)
            if uses_only_flex:
                pad_mask = None
                key_padding_mask = None
        needs_dual_flex_mask = any(
            space == "dual" and backend == "flex"
            for space, backend in zip(
                self.config.attention_spaces,
                self.config.attention_backends,
            )
        )
        block_mask, dual_attention_mask = _prepare_document_masks(
            flex_document_ids,
            include_dense_mask=needs_dual_flex_mask,
            padding_only=document_ids is None,
        )

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.embedding_norm(self.encoder(src))

        # Positional embedding
        if not self.config.rope:
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x = layer(
                x,
                pad_mask,
                freqs_cis,
                key_padding_mask,
                block_mask,
                dual_attention_mask,
            )

        # Final normalization layer
        x = self.layer_norm(x)

        # Return the output of the last hidden layer
        return x


class NormNeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        else:
            self.positional_embedding = nn.Embedding(config.max_length + 1, config.hidden_size, padding_idx=config.pad_token_id)

        self.transformer_encoder = nn.ModuleList()
        for layer_index in range(config.num_hidden_layers):
            self.transformer_encoder.append(NormEncoderBlock(config, layer_index))

        self.layer_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.num_hidden_layers))

        self.sz_init_value = 1.00
        self.sz_init_scaling = config.base_scale
        self.sz = torch.nn.Parameter(self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32))

    def forward(self, src, pad_mask=None, document_ids=None):
        if document_ids is not None:
            raise ValueError("packed document masking is unavailable for ngpt=True")
        uses_flex = any(
            backend == "flex" for backend in self.config.attention_backends
        )
        uses_only_flex = all(
            backend == "flex" for backend in self.config.attention_backends
        )
        pad_mask, key_padding_mask = _prepare_attention_masks(
            pad_mask,
            self.config.num_attention_heads,
            src.shape[1],
        )
        flex_document_ids = None
        if key_padding_mask is not None and uses_flex:
            flex_document_ids = _document_ids_from_key_padding_mask(key_padding_mask)
            if uses_only_flex:
                pad_mask = None
                key_padding_mask = None
        block_mask, _ = _prepare_document_masks(
            flex_document_ids,
            include_dense_mask=False,
            padding_only=True,
        )

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x = layer(x, pad_mask, freqs_cis, key_padding_mask, block_mask)

        # Return the output of the last hidden layer
        return x


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.model = NormNeoBERT(config) if self.config.ngpt else NeoBERT(config)
        self.decoder = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=config.lm_head_bias,
        )

        self.post_init()
        if config.tie_word_embeddings:
            self.decoder.weight = self.model.encoder.weight

    def forward(self, src, pad_mask=None, document_ids=None):
        hidden_representation = self.model.forward(src, pad_mask, document_ids)
        logits = self.decoder(hidden_representation)

        return {"hidden_representation": hidden_representation, "logits": logits}


class NeoBERTForSequenceClassification(NeoBERTPreTrainedModel):

    def __init__(
        self,
        config: NeoBERTConfig,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_init_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config

        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.classifier_init_range = classifier_init_range

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, src, pad_mask=None):
        hidden_representation = self.model.forward(src, pad_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        return {"hidden_representation": hidden_representation, "logits": logits}


class NeoBERTHFForSequenceClassification(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        hidden_representation = self.model.forward(input_ids, attention_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )


class NeoBERTForMTEB(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(
        self,
        config: NeoBERTConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        batch_size: int = 8,
        pooling: str = "avg",
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        self.model = NeoBERT(config)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

    def encode_queries(self, queries: List[str], **kwargs):
        if "instructions" in kwargs:
            if kwargs["instructions"] is not None:
                queries = [(query + " " + kwargs["instructions"][query]).strip() for query in queries]
            new_kwargs = {k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]}
        else:
            new_kwargs = kwargs

        return self.encode(
            queries,
            **new_kwargs,
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            if isinstance(corpus[0], dict):
                sentences = [(doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
            else:
                sentences = corpus

        if "instructions" in kwargs:  # not used on the doc side
            new_kwargs = {k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]}
        else:
            new_kwargs = kwargs

        return self.encode(
            sentences,
            **new_kwargs,
        )

    @torch.no_grad()
    def encode(self, sentences: list[str], **kwargs: Any) -> torch.Tensor:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _transform_func(tokenizer: PreTrainedTokenizerFast, x: Dict[str, List]):
            batch_dict = tokenizer(
                x["input_texts"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_token_type_ids=False,
            )

            return batch_dict

        dataset: Dataset = Dataset.from_dict({"input_texts": sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            collate_fn=data_collator,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
        )

        encodings = []
        for batch in tqdm(dataloader, desc="encoding", mininterval=10, disable=len(sentences) < 128):
            input_ids = batch["input_ids"].to(device)

            pad_mask = batch["attention_mask"].to(device)
            xformers_mask = torch.where(pad_mask == 1, float(0.0), float("-inf")).type(torch.float16)

            outputs = self.model(input_ids, xformers_mask)

            if self.pooling == "avg":
                outputs = outputs * pad_mask.unsqueeze(-1).expand(-1, -1, outputs.shape[-1])
                outputs = outputs.sum(dim=1) / pad_mask.to(device).sum(dim=1).unsqueeze(-1)
            else:
                outputs = outputs[:, 0, :]

            encodings.append(outputs.cpu().numpy())

        return np.concatenate(encodings, axis=0)
