"""Compatibility hook for official BabyLM fine-tuning of NeoBERT exports.

The official pipeline loads an encoder through ``AutoModel`` and supplies
padded FP32 batches.  NeoBERT exports register the masked-LM auto class, while
their direct Flash kernels intentionally require padding-free BF16 inputs.
For supervised evaluation only, load the tied masked-LM checkpoint, retain its
encoder, and select the algebraically equivalent PyTorch attention backend.
This changes no learned tensors and permits the official padded FP32 recipe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _is_local_neobert(path_or_name) -> bool:
    try:
        config_path = Path(path_or_name) / "config.json"
    except TypeError:
        return False
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return config.get("model_type") == "neobert"


def _select_torch_attention(model) -> None:
    config = model.config
    config.attention_backend = "torch"
    config.attention_backends = ["torch"] * int(config.num_hidden_layers)
    for module in model.modules():
        if getattr(module, "attention_backend", None) in ("flash", "flash_fused"):
            module.attention_backend = "torch"
        if getattr(module, "backend", None) in ("flash", "flash_fused"):
            module.backend = "torch"


if os.environ.get("NEOBERT_BABYLM_FINETUNE_COMPAT") == "1":
    import torch
    from transformers import AutoModel, AutoModelForMaskedLM
    from transformers.modeling_outputs import BaseModelOutput

    class _NeoBERTEncoderAdapter(torch.nn.Module):
        """Expose NeoBERT's compact encoder through the HF base-model API."""

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.config = encoder.config

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            if input_ids is None:
                raise ValueError("input_ids is required")
            if inputs_embeds is not None or position_ids is not None:
                raise ValueError("NeoBERT does not accept external embeddings or positions")
            if kwargs:
                raise TypeError(f"unsupported NeoBERT encoder arguments: {sorted(kwargs)}")
            del token_type_ids, output_attentions
            hidden = self.encoder(input_ids, attention_mask)
            if return_dict is False:
                return (hidden,)
            return BaseModelOutput(
                last_hidden_state=hidden,
                hidden_states=(hidden,) if output_hidden_states else None,
                attentions=None,
            )

    seed = int(os.environ.get("NEOBERT_BABYLM_SEED", "42"))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _original_auto_model_from_pretrained = AutoModel.from_pretrained

    def _from_pretrained_neobert(cls, pretrained_model_name_or_path, *args, **kwargs):
        if not _is_local_neobert(pretrained_model_name_or_path):
            return _original_auto_model_from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        # Loading the two architecture variants constructs different module
        # trees and would otherwise advance RNG by different amounts before
        # the official classifier head is initialized.  Isolate that loading
        # so a shared NEOBERT_BABYLM_SEED produces an exactly paired head.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            masked_lm = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
            encoder = masked_lm.model
            _select_torch_attention(encoder)
            if next(encoder.parameters()).dtype != torch.float32:
                encoder.float()
            return _NeoBERTEncoderAdapter(encoder)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

    AutoModel.from_pretrained = classmethod(_from_pretrained_neobert)
