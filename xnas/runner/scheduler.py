"""Learning rate schedulers."""

import math
import torch
from xnas.core.config import cfg

from torch.optim.lr_scheduler import _LRScheduler


__all__ = ['lr_scheduler_builder', 'adjust_learning_rate_per_batch']


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


def _calc_learning_rate(
    init_lr, n_epochs, epoch, n_iter=None, iter=0,
):
    epoch -= cfg.OPTIM.WARMUP_EPOCH
    if cfg.OPTIM.LR_POLICY == "cos":
        t_total = n_epochs * n_iter
        t_cur = epoch * n_iter + iter
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif cfg.OPTIM.LR_POLICY == "step":
        # Rule of BigNAS: decay learning rate by 0.97 every 2.4 epochs
        # t_total = n_epochs * n_iter
        # t_cur = epoch * n_iter + iter
        t_cur_epoch = epoch + iter / n_iter
        lr = (0.97 ** (t_cur_epoch / 2.4)) * init_lr
    else:
        raise ValueError("do not support: {}".format(cfg.OPTIM.LR_POLICY))
    return lr


def _warmup_adjust_learning_rate(
        init_lr, n_epochs, epoch, n_iter, iter=0, warmup_lr=0
    ):
        """adjust lr during warming-up. Changes linearly from `warmup_lr` to `init_lr`."""
        T_cur = epoch * n_iter + iter + 1
        t_total = n_epochs * n_iter
        new_lr = T_cur / t_total * (init_lr - warmup_lr) + warmup_lr
        return new_lr


def adjust_learning_rate_per_batch(epoch, n_iter=None, iter=0, warmup=False):
    """adjust learning of a given optimizer and return the new learning rate"""
    
    init_lr = cfg.OPTIM.BASE_LR * cfg.NUM_GPUS
    n_epochs = cfg.OPTIM.MAX_EPOCH
    n_warmup_epochs = cfg.OPTIM.WARMUP_EPOCH
    warmup_lr = init_lr * cfg.OPTIM.WARMUP_FACTOR
    
    if warmup:
        new_lr = _warmup_adjust_learning_rate(
            init_lr, n_warmup_epochs, epoch, n_iter, iter, warmup_lr
        )
    else:
        new_lr = _calc_learning_rate(
            init_lr, n_epochs, epoch, n_iter, iter
        )
    return new_lr
