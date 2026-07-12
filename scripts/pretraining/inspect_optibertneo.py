import argparse
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from neobert.model import NeoBERTConfig, NeoBERTLMHead


PAPER_NON_EMBEDDING_PARAMETERS = 12 * 768**2 * 28
MODEL_CONFIGS = {
    "baseline": "optibertneo-198m.yaml",
    "mixed": "optibertneo-mixed-198m.yaml",
}


def unique_parameter_count(parameters):
    unique = {id(parameter): parameter for parameter in parameters}
    return sum(parameter.numel() for parameter in unique.values())


def inspect_variant(variant):
    config_dir = Path(__file__).resolve().parents[2] / "conf" / "model"
    values = OmegaConf.to_container(OmegaConf.load(config_dir / MODEL_CONFIGS[variant]))
    config = NeoBERTConfig(vocab_size=50265, pad_token_id=1, max_length=1024, **values)
    model = NeoBERTLMHead(config)

    embedding = model.model.encoder.weight
    total = unique_parameter_count(model.parameters())
    non_embedding = unique_parameter_count(
        parameter for parameter in model.parameters() if parameter is not embedding
    )
    relative_error = (non_embedding - PAPER_NON_EMBEDDING_PARAMETERS) / PAPER_NON_EMBEDDING_PARAMETERS
    spaces = Counter(config.attention_spaces)
    print(f"variant: {variant}")
    print(f"hidden size: {config.hidden_size}")
    print(f"heads: {config.num_attention_heads} x {config.dim_head}")
    print(f"attention spaces: {dict(spaces)}")
    print(f"non-embedding parameters: {non_embedding:,}")
    print(f"paper target: {PAPER_NON_EMBEDDING_PARAMETERS:,}")
    print(f"relative difference: {relative_error:+.4%}")
    print(f"unique total parameters: {total:,}")
    return abs(relative_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("baseline", "mixed", "all"), default="all", nargs="?")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    variants = MODEL_CONFIGS if args.variant == "all" else (args.variant,)
    errors = []
    for index, variant in enumerate(variants):
        if index:
            print()
        errors.append(inspect_variant(variant))
    if args.check and any(error > 0.001 for error in errors):
        raise SystemExit("a model differs from the paper parameter target by more than 0.1%")


if __name__ == "__main__":
    main()
