# NeoBERT

## Description

NeoBERT is a **next-generation encoder** model for English text representation, pre-trained from scratch on the RefinedWeb dataset. NeoBERT integrates state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. It is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an **optimal depth-to-width ratio**, and leverages an extended context length of **4,096 tokens**. Despite its compact 250M parameter footprint, it is the most efficient model of its kind and achieves **state-of-the-art results** on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions. 

- Paper: [paper](https://arxiv.org/abs/2502.19587)
- Model: [huggingface](https://huggingface.co/chandar-lab/NeoBERT).

## Get started

Ensure you have the following dependencies installed:

```bash
pip install transformers torch xformers==0.0.28.post3
```

xFormers integration is limited to the fused SwiGLU implementation; it is not
an attention backend. Sequence packing uses the FlexAttention API included in
PyTorch. Attention backends are selected per scalar space:

| Scalar space | Supported backends |
| --- | --- |
| Real | `auto`, `torch`, `flash`, `flex` |
| Ordinary complex | `auto`, `native`, `torch`, `flash`, `flex` |
| Split complex | `auto`, `native`, `torch`, `flash`, `flex` |
| Dual number | `auto`, `native`, `torch`, `flash`, `flash_fused`, `flex` |

The legacy `flash_attention` flag maps `true` to `flash` and `false` to
`torch`. Packed-document schedules should use `flex`; direct `flash` does not
support arbitrary document masks. Split-complex `flash` uses a single packed
half-idempotent Flash SDPA call and requires `attention_dropout=0` so both
channels retain the same attention semantics. Dual-number `flash` is a strict
hybrid: Flash SDPA computes the primal output, while the exact tangent is
recomputed densely and therefore still uses quadratic attention memory. It is
CUDA FP16/BF16 only, requires equal Q/K/V head widths no greater than 256,
square causal attention or no mask, no actually padded keys, no attention
weights, and `attention_dropout=0`. Use `dual` with `flex` for packed-document
masks; its exact tangent is likewise dense, but it supports attention dropout.
For linear-memory dual attention, `flash_fused` streams the primal and tangent
through Triton kernels. It requires finite CUDA FP16/BF16 inputs, Triton, an
SM80-or-newer GPU, head dimensions no larger than 128, no padded keys or
arbitrary masks, and `attention_dropout=0`; its custom backward supports
first-order reverse-mode autograd.

## How to use

Load the model using Hugging Face Transformers:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "chandar-lab/NeoBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Tokenize input text
text = "NeoBERT is the most efficient model of its kind!"
inputs = tokenizer(text, return_tensors="pt")

# Generate embeddings
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]
print(embedding.shape)
```

## Features
| **Feature**       | **NeoBERT**                             |
|---------------------------|-----------------------------|
| `Depth-to-width`        | 28 × 768  |
| `Parameter count`           | 250M                        |
| `Activation`               | SwiGLU                      |
| `Positional embeddings`     | RoPE                        |
| `Normalization`            | Pre-RMSNorm                 |
| `Data Source`              | RefinedWeb                  |
| `Data Size`                | 2.8 TB                       |
| `Tokenizer`                | google/bert                 |
| `Context length`    | 4,096                       |
| `MLM Masking Rate`             | 20%                         |
| `Optimizer`                | AdamW                       |
| `Scheduler`                | CosineDecay                 |
| `Training Tokens`          | 2.1 T                        |
| `Efficiency`               | FlashAttention              |

## License

Model weights and code repository are licensed under the permissive MIT license.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{breton2025neobertnextgenerationbert,
      title={NeoBERT: A Next-Generation BERT}, 
      author={Lola Le Breton and Quentin Fournier and Mariam El Mezouar and Sarath Chandar},
      year={2025},
      eprint={2502.19587},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.19587}, 
}
```

## Contact

For questions, do not hesitate to reach out and open an issue on here or on our **[GitHub](https://github.com/chandar-lab/NeoBERT)**.

---
