"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from xnas.core.config import cfg


__all__ = ['criterion_builder']


def _label_smooth(target, n_classes: int, label_smoothing):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target


def CrossEntropyLoss_soft_target(pred, soft_target):
    """CELoss with soft target, mainly used during KD"""
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), dim=1))


def CrossEntropyLoss_label_smoothed(pred, target, label_smoothing=0.):
    label_smoothing = cfg.SEARCH.LABEL_SMOOTH if label_smoothing == 0. else label_smoothing
    soft_target = _label_smooth(target, pred.size(1), label_smoothing)
    return CrossEntropyLoss_soft_target(pred, soft_target)


class MultiHeadCrossEntropyLoss(nn.Module):
    def forward(self, preds, targets):
        assert preds.dim() == 3, preds
        assert targets.dim() == 2, targets

        assert preds.size(1) == targets.size(1), (preds, targets)
        num_heads = targets.size(1)

        loss = 0
        for k in range(num_heads):
            loss += F.cross_entropy(preds[:, k, :], targets[:, k]) / num_heads
        return loss


# ----------

SUPPORTED_CRITERIONS = {
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    "cross_entropy_smooth": CrossEntropyLoss_label_smoothed,
    "cross_entropy_multihead": MultiHeadCrossEntropyLoss()
}


def criterion_builder():
    err_str = "Loss function type '{}' not supported"
    assert cfg.SEARCH.LOSS_FUN in SUPPORTED_CRITERIONS.keys(), err_str.format(cfg.SEARCH.LOSS_FUN)
    return SUPPORTED_CRITERIONS[cfg.SEARCH.LOSS_FUN]
