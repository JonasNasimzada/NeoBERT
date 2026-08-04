import torch
from torch.optim import AdamW, Adam

from accelerate.utils import DistributedType


def get_optimizer(model: torch.nn.Module, distributed_type: DistributedType, **kwargs) -> torch.optim.Optimizer:
    """Optimizer.

    Args:
        model (torch.nn.Module): Model.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    match kwargs.pop("name"):
        case "AdamW":
            return AdamW(model.parameters(), **kwargs)
        case "Adam":
            return Adam(model.parameters(), **kwargs)
        case "SOAP":
            assert distributed_type is not DistributedType.DEEPSPEED, "SOAP does not support DeepSpeed"
            try:
                from .soap.soap import SOAP
            except ModuleNotFoundError as error:
                if error.name and error.name.startswith(f"{__package__}.soap"):
                    raise ImportError(
                        "SOAP was selected, but the optional neobert.optimizer.soap "
                        "implementation is not installed"
                    ) from error
                raise
            return SOAP(model.parameters(), **kwargs)
        case _:
            raise ValueError("Unrecognized optimizer name. Options are: Adam, AdamW, SOAP.")
