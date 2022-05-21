"""Optimizers and loss functions."""

import torch
from xnas.core.config import cfg


__all__ = [
    'optimizer_builder', 
    'darts_alpha_optimizer', 
    'criterion_builder'
]


SUPPORTED_OPTIMIZERS = {
    "SGD",
    "Adam",
}

SUPPORTED_CRITERIONS = {
    "cross_entropy": torch.nn.CrossEntropyLoss(),
}


def optimizer_builder(name, param):
    """optimizer builder

    Args:
        name (str): name of optimizer
        param (dict): parameters to optimize

    Returns:
        optimizer: optimizer
    """
    assert name in SUPPORTED_OPTIMIZERS, "optimizer not supported."
    if name == "SGD":
        return torch.optim.SGD(
            param,
            cfg.OPTIM.BASE_LR,
            cfg.OPTIM.MOMENTUM,
            cfg.OPTIM.DAMPENING,    # 0.0 following default
            cfg.OPTIM.WEIGHT_DECAY,
            cfg.OPTIM.NESTEROV,     # False following default
        )
    elif name == "Adam":
        return torch.optim.Adam(
            param,
            cfg.OPTIM.BASE_LR,
            betas=(0.5, 0.999),
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )


def darts_alpha_optimizer(name, param):
    """alpha optimizer for DARTS-like methods.
    Make sure cfg.DARTS has been initialized.

    Args:
        name (str): name of optimizer
        param (dict): parameters to optimize

    Returns:
        optimizer: optimizer
    """
    assert name in SUPPORTED_OPTIMIZERS, "optimizer not supported."
    if name == "Adam":
        return torch.optim.Adam(
            param,
            cfg.DARTS.ALPHA_LR,
            betas=(0.5, 0.999),
            weight_decay=cfg.DARTS.ALPHA_WEIGHT_DECAY,
        )


def criterion_builder():
    err_str = "Loss function type '{}' not supported"
    assert cfg.SEARCH.LOSS_FUN in SUPPORTED_CRITERIONS.keys(), err_str.format(cfg.SEARCH.LOSS_FUN)
    return SUPPORTED_CRITERIONS[cfg.SEARCH.LOSS_FUN]
