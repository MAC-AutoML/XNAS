# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

import torch

from xnas.core.config import cfg
from xnas.core.warmup_sheduler import GradualWarmupScheduler
from xnas.search_algorithm.ASNG import ASNG, Dynamic_ASNG
from xnas.search_algorithm.DDPNAS import CategoricalDDPNAS
from xnas.search_algorithm.MDENAS import CategoricalMDENAS
from xnas.search_algorithm.MIGO import MIGO
# MIGO series
from xnas.search_algorithm.SNG import SNG, Dynamic_SNG
# DARTS series space
from xnas.search_space.DARTS.cnn import _DartsCNN
# DrNAS series modified space
from xnas.search_space.DrNAS.DARTSspace.cnn import _DrNASCNN_DARTSspace
from xnas.search_space.DrNAS.nb201space.cnn import _DrNASCNN_nb201space, _DrNASCNN_GDAS_nb201space
from xnas.search_space.NASBench1shot1.cnn import _NASbench1shot1_1, _NASbench1shot1_2, _NASbench1shot1_3
# NAS-Bench series space
from xnas.search_space.NASBench201.cnn import _NASBench201
# OFA series space
from xnas.search_space.OFA.ofa_networks import _OFAMobileNetV3, _OFAProxylessNASNet, _OFAResNet
from xnas.search_space.OFA.utils import (cross_entropy_loss_with_label_smoothing, cross_entropy_loss_with_soft_target)
from xnas.search_space.PCDARTS.cnn import _PcdartsCNN
from xnas.search_space.PDARTS.cnn import _PdartsCNN
# SPOS space
from xnas.search_space.SPOS.supernet import _SPOSSUPNET

# Supported models
_spaces = {
    "darts": _DartsCNN,
    "pdarts": _PdartsCNN,
    "pcdarts": _PcdartsCNN,
    "ofa_mbv3": _OFAMobileNetV3,
    "ofa_proxyless": _OFAProxylessNASNet,
    "ofa_resnet": _OFAResNet,
    "nasbench1shot1_1": _NASbench1shot1_1,
    "nasbench1shot1_2": _NASbench1shot1_2,
    "nasbench1shot1_3": _NASbench1shot1_3,
    "nasbench201": _NASBench201,
    "nasbench301": _DartsCNN,
    "spos": _SPOSSUPNET
}

# Supported loss functions
_loss_funs = {
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    "cross_entropy_with_label_smoothing": cross_entropy_loss_with_label_smoothing,
    "cross_entropy_with_soft_target": cross_entropy_loss_with_soft_target,
}


def get_space():
    """Get the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.SPACE.NAME in _spaces.keys(), err_str.format(cfg.SPACE.NAME)
    return _spaces[cfg.SPACE.NAME]


def get_loss_fun():
    """Get the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    # using cfg.SEARCH space instead
    # assert cfg.TRAIN.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS_FUN)
    # return _loss_funs[cfg.TRAIN.LOSS_FUN]
    assert cfg.SEARCH.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.SEARCH.LOSS_FUN)
    return _loss_funs[cfg.SEARCH.LOSS_FUN]


def build_space():
    """Build the model."""
    return get_space()()  # TODO: add **kwargs for it.


def build_loss_fun():
    """Build the loss function."""
    return get_loss_fun()


def register_space(name, ctor):
    """Register a model dynamically."""
    _spaces[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor


def DrNAS_builder():
    criterion = build_loss_fun()
    if cfg.SPACE.NAME == 'darts':
        return _DrNASCNN_DARTSspace(criterion)
    elif cfg.SPACE.NAME == 'nasbench201':
        if cfg.DRNAS.METHOD == 'gdas':
            return _DrNASCNN_GDAS_nb201space(criterion)
        elif cfg.DRNAS.METHOD == 'snas':
            return _DrNASCNN_nb201space('gumbel', criterion)
        elif cfg.DRNAS.METHOD == 'dirichlet':
            return _DrNASCNN_nb201space('dirichlet', criterion)
        elif cfg.DRNAS.METHOD == 'darts':
            return _DrNASCNN_nb201space('softmax', criterion)
        else:
            raise NotImplementedError


def sng_builder(category):
    if cfg.SNG.NAME == 'SNG':
        return SNG(category, lam=cfg.SNG.LAMBDA)
    elif cfg.SNG.NAME == 'ASNG':
        return ASNG(category, lam=cfg.SNG.LAMBDA)
    elif cfg.SNG.NAME == 'dynamic_SNG':
        return Dynamic_SNG(category, step=cfg.SNG.PRUNING_STEP, pruning=cfg.SNG.PRUNING)
    elif cfg.SNG.NAME == 'dynamic_ASNG':
        return Dynamic_ASNG(category, step=cfg.SNG.PRUNING_STEP, pruning=cfg.SNG.PRUNING)
    elif cfg.SNG.NAME == 'MDENAS':
        return CategoricalMDENAS(category, cfg.SNG.THETA_LR)
    elif cfg.SNG.NAME == 'DDPNAS':
        return CategoricalDDPNAS(category, cfg.SNG.PRUNING_STEP)
    elif cfg.SNG.NAME == 'MIGO':
        return MIGO(categories=category,
                    step=cfg.SNG.PRUNING_STEP, lam=cfg.SNG.LAMBDA,
                    pruning=cfg.SNG.PRUNING, sample_with_prob=cfg.SNG.PROB_SAMPLING,
                    utility_function=cfg.SNG.UTILITY, utility_function_hyper=cfg.SNG.UTILITY_FACTOR,
                    momentum=cfg.SNG.MOMENTUM, gamma=cfg.SNG.GAMMA, sampling_number_per_edge=cfg.SNG.SAMPLING_PER_EDGE)
    else:
        raise NotImplementedError


def lr_scheduler_builder(w_optim, last_epoch=-1):
    if cfg.OPTIM.LR_POLICY == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, cfg.OPTIM.MAX_EPOCH, eta_min=cfg.OPTIM.MIN_LR,
                                                          last_epoch=last_epoch)
    elif cfg.OPTIM.LR_POLICY == "step":
        return torch.optim.lr_scheduler.MultiStepLR(w_optim, cfg.OPTIM.STEPS, gamma=cfg.OPTIM.LR_MULT,
                                                    last_epoch=last_epoch)
    else:
        raise NotImplementedError


def warmup_scheduler_builder(w_optim, actual_scheduler, last_epoch=-1):
    if cfg.OPTIM.WARMUP_EPOCHS > 0:
        return GradualWarmupScheduler(w_optim, actual_scheduler, cfg.OPTIM.WARMUP_EPOCHS, cfg.OPTIM.WARMUP_FACTOR,
                                      last_epoch)
    else:
        return actual_scheduler
