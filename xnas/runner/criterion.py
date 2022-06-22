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


class KLLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification 
            output: output logits of the student network
            target: output logits of the teacher network
            T: temperature
            KL(p||q) = Ep \log p - \Ep log q
    """
    def forward(self, output, soft_logits, target=None, temperature=1., alpha=0.9):
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = F.softmax(soft_logits, dim=1)
        output_log_prob = F.log_softmax(output, dim=1)
        kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
        if target is not None:
            n_class = output.size(1)
            target = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
            target = target.unsqueeze(1)
            output_log_prob = output_log_prob.unsqueeze(2)
            ce_loss = -torch.bmm(target, output_log_prob).squeeze()
            loss = alpha * temperature * temperature * kd_loss + (1.0 - alpha) * ce_loss
        else:
            loss = kd_loss 
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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
    "cross_entropy_soft": CrossEntropyLoss_soft_target,
    "cross_entropy_smooth": CrossEntropyLoss_label_smoothed,
    "cross_entropy_multihead": MultiHeadCrossEntropyLoss(),
    "kl_soft": KLLossSoft(),
}


def criterion_builder(criterion=None):
    err_str = "Loss function type '{}' not supported"
    loss_fun = cfg.SEARCH.LOSS_FUN if criterion is None else criterion
    assert loss_fun in SUPPORTED_CRITERIONS.keys(), err_str.format(loss_fun)
    return SUPPORTED_CRITERIONS[loss_fun]
