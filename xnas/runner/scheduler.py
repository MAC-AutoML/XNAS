"""Learning rate schedulers."""

import torch
from xnas.core.config import cfg

from torch.optim.lr_scheduler import _LRScheduler


__all__ = ['lr_scheduler_builder']


def lr_scheduler_builder(optimizer, last_epoch=-1, **kwargs):
    """Learning rate scheduler, now support warmup_epoch."""
    actual_scheduler = None
    if cfg.OPTIM.LR_POLICY == "cos":
        actual_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs['T_max'] if 'T_max' in kwargs.keys() else cfg.OPTIM.MAX_EPOCH,
            eta_min=cfg.OPTIM.MIN_LR,
            last_epoch=last_epoch)
    elif cfg.OPTIM.LR_POLICY == "step":
        actual_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.OPTIM.STEPS,
            gamma=cfg.OPTIM.LR_MULT,
            last_epoch=last_epoch)
    else:
        raise NotImplementedError

    if cfg.OPTIM.WARMUP_EPOCH > 0:
        return GradualWarmupScheduler(
            optimizer,
            actual_scheduler,
            cfg.OPTIM.WARMUP_EPOCH,
            cfg.OPTIM.WARMUP_FACTOR, 
            last_epoch)
    else:
        return actual_scheduler


class GradualWarmupScheduler(_LRScheduler):
    """
    Implementation Reference:
        https://github.com/ildoonet/pytorch-gradual-warmup-lr
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor: start the warm up from init_lr * factor
        actual_scheduler: after warmup_epochs, use this scheduler
        warmup_epochs: init_lr is reached at warmup_epochs, linearly
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 actual_scheduler: _LRScheduler,
                 warmup_epochs: int,
                 factor: float,
                 last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        
        self.actual_scheduler = actual_scheduler
        self.warmup_epochs = warmup_epochs
        self.factor = factor
        self.last_epoch = last_epoch


    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            return self.actual_scheduler.get_lr()
        else:
            return [base_lr * ((1. - self.factor) * self.last_epoch / self.warmup_epochs + self.factor) for base_lr in self.base_lrs]


    def step(self, epoch=None):
        if self.last_epoch > self.warmup_epochs:
            if epoch is None:
                self.actual_scheduler.step(None)
            else:
                self.actual_scheduler.step(epoch - self.warmup_epochs)
            self._last_lr = self.actual_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
