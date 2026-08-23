"""Fail-fast A100 and backend smoke check for an ablation array task."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from neobert.model import NeoBERTConfig, NeoBERTLMHead


VALID_BACKENDS = {
    "real": {"torch", "flash"},
    "complex": {"native", "torch", "flash"},
    "split": {"native", "torch", "flash"},
    "dual": {"native", "torch", "flash", "flash_fused"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, choices=sorted(VALID_BACKENDS))
    parser.add_argument("--backend", required=True)
    parser.add_argument("--require-a100", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend not in VALID_BACKENDS[args.space]:
        raise ValueError(f"unsupported pair: {args.space}/{args.backend}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    if args.require_a100 and "A100" not in device_name.upper():
        raise RuntimeError(f"expected an A100, found {device_name}")
    if args.require_a100 and capability != (8, 0):
        raise RuntimeError(f"expected SM80, found compute capability {capability}")
    if "sm_80" not in arch_list:
        raise RuntimeError(
            "this PyTorch/CUDA build does not contain sm_80 code; rebuild with "
            "TORCH_CUDA_ARCH_LIST=8.0"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    config = NeoBERTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        vocab_size=128,
        max_length=16,
        rope=True,
        rms_norm=True,
        embedding_rms_norm=False,
        tie_word_embeddings=True,
        lm_head_bias=False,
        dropout=0.0,
        attention_dropout=0.0,
        attention_space=args.space,
        attention_backend=args.backend,
    )
    model = NeoBERTLMHead(config).cuda().train()
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device="cuda")
    labels = torch.randint(0, config.vocab_size, (2, 16), device="cuda")
    # FlexAttention requires a BlockMask even for an otherwise unmasked smoke
    # batch.  A single document id per row exercises the same model-level mask
    # construction used by the packed training dataloader.
    document_ids = (
        torch.zeros_like(input_ids, dtype=torch.int32)
        if args.backend == "flex"
        else None
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids, document_ids=document_ids)["logits"]
        loss = F.cross_entropy(
            logits.float().reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("backend smoke loss is not finite")
    loss.backward()
    if not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise RuntimeError("backend smoke produced a non-finite gradient")
    torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "space": args.space,
                "backend": args.backend,
                "device": device_name,
                "capability": capability,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "arch_list": arch_list,
                "bf16": True,
                "smoke_loss": float(loss.detach()),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
