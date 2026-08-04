"""Padding-free inference adapter for strict FlashAttention checkpoints.

The BabyLM zero-shot harness pads variable-length MLM examples before calling
the model.  The strict Flash backend intentionally rejects padded keys.  This
adapter keeps the checkpoint's backend unchanged and evaluates each padded row
at its true length, then restores the batch-shaped logits expected by the
harness.  It is installed only inside the optional evaluator subprocess.
"""

from __future__ import annotations

import types

import torch
from torch.nn import functional as F
from transformers.modeling_outputs import MaskedLMOutput


def _uses_strict_flash(model) -> bool:
    backends = getattr(model.config, "attention_backends", None)
    if backends is None:
        backends = [getattr(model.config, "attention_backend", None)]
    return "flash" in backends


def install_padding_free_flash_forward(model):
    """Install a row-wise unpadding shim when a checkpoint selects Flash."""
    if not _uses_strict_flash(model):
        return model

    original_forward = model.forward

    def padding_free_forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        if input_ids is None or attention_mask is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        if bool(attention_mask.bool().all()):
            return original_forward(
                input_ids=input_ids,
                attention_mask=None,
                **kwargs,
            )
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "the BabyLM Flash adapter expects input_ids and attention_mask "
                "with the same (batch, sequence) shape"
            )
        if kwargs.get("labels") is not None:
            raise ValueError(
                "the BabyLM Flash adapter is inference-only and does not accept labels"
            )

        batch_size, padded_length = input_ids.shape
        padded_logits = []
        for row_index in range(batch_size):
            valid = attention_mask[row_index].bool()
            true_length = int(valid.sum().item())
            if true_length <= 0:
                raise ValueError("the BabyLM evaluator produced an empty sequence")
            if not bool(valid[:true_length].all()) or bool(valid[true_length:].any()):
                raise ValueError(
                    "the BabyLM Flash adapter supports right padding only"
                )

            row_kwargs = {}
            for name, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                    value = value[row_index : row_index + 1]
                    if value.ndim >= 2 and value.shape[1] == padded_length:
                        value = value[:, :true_length]
                row_kwargs[name] = value

            output = original_forward(
                input_ids=input_ids[row_index : row_index + 1, :true_length],
                attention_mask=None,
                **row_kwargs,
            )
            logits = output.logits if hasattr(output, "logits") else output[0]
            padded_logits.append(
                F.pad(logits, (0, 0, 0, padded_length - true_length))
            )

        return MaskedLMOutput(logits=torch.cat(padded_logits, dim=0))

    model.forward = types.MethodType(padding_free_forward, model)
    return model
