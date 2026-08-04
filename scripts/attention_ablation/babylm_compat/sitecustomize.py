"""Narrow compatibility hook for the official BabyLM evaluator.

The upstream zero-shot runner does not expose a model dtype argument. Strict
Flash SDPA on A100 rejects its default FP32 load, so the Slurm benchmark job
enables this hook only for evaluator subprocesses.
"""

from __future__ import annotations

import os


if os.environ.get("NEOBERT_BABYLM_FORCE_BF16") == "1":
    import torch
    from transformers import AutoModelForMaskedLM

    from padding_free_flash import install_padding_free_flash_forward

    _original_from_pretrained = AutoModelForMaskedLM.from_pretrained

    def _from_pretrained_bf16(cls, *args, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        model = _original_from_pretrained(*args, **kwargs)
        return install_padding_free_flash_forward(model)

    AutoModelForMaskedLM.from_pretrained = classmethod(_from_pretrained_bf16)
