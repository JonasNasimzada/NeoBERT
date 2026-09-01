"""Hydra entry point for the controlled, equal-token attention ablation."""

import hydra
from omegaconf import DictConfig

from neobert.pretraining import trainer


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="attention_ablation",
)
def pipeline(cfg: DictConfig) -> None:
    trainer(cfg)


if __name__ == "__main__":
    pipeline()
