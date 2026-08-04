import math
from collections import defaultdict

import torch
from accelerate import Accelerator


class Metrics(defaultdict):

    def __init__(self):
        super().__init__(int)

    def state_dict(self):
        return dict(self)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            self[k] = v

    def log(
        self,
        accelerator: Accelerator,
        *,
        step: int | None = None,
        extra_metrics: dict | None = None,
    ):
        """Aggregate interval counters and emit one flat tracker record.

        Only ``*/local_*`` counters are rank-local.  Keeping them out of the
        tracker payload avoids exposing main-rank implementation details and
        also avoids needless reductions of global counters such as the
        optimizer step.
        """
        local_keys = tuple(key for key in self if "/local_" in key)
        if local_keys:
            metrics_agg = torch.tensor(
                [self[key] for key in local_keys],
                dtype=torch.float64,
                device=accelerator.device,
            )
            metrics_agg = (
                accelerator.reduce(metrics_agg, reduction="sum")
                .detach()
                .cpu()
                .tolist()
            )
            metrics_agg = dict(zip(local_keys, metrics_agg))
        else:
            metrics_agg = {}

        # Update global values
        self["train/samples"] += int(
            metrics_agg.get("train/local_samples", 0)
        )
        self["train/tokens"] += int(
            metrics_agg.get("train/local_tokens", 0)
        )
        self["train/masked_tokens"] += int(
            metrics_agg.get("train/local_num_pred", 0)
        )

        # Build the metrics to log
        metrics_log = {
            key: value
            for key, value in self.items()
            if "/local_" not in key
        }

        local_num_pred = metrics_agg.get("train/local_num_pred", 0)
        if local_num_pred > 0:
            metrics_log["train/loss"] = (
                metrics_agg["train/local_sum_loss"] / local_num_pred
            )
            metrics_log["train/perplexity"] = math.exp(
                metrics_log["train/loss"]
            )
            metrics_log["train/accuracy"] = (
                metrics_agg["train/local_num_correct"] / local_num_pred
            )

        if extra_metrics:
            metrics_log.update(extra_metrics)

        # Log the metrics
        accelerator.log(metrics_log, step=step)

        # Reset the local counters
        for key in local_keys:
            self.pop(key, None)

        return metrics_log
