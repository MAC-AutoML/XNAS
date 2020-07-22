#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

import torch
from xnas.core.config import cfg
from xnas.search_space.cell_based import _DartsCNN
from xnas.search_space.cell_based import _NASbench201
<<<<<<< HEAD
from xnas.search_space.method.pdarts import _PdartsCNN
from xnas.search_space.mb_v3_cnn import build_super_net
from xnas.search_algorithm.ASNG import ASNG, Dynamic_ASNG
from xnas.search_algorithm.SNG import SNG, Dynamic_SNG
from xnas.search_algorithm.MDENAS import CategoricalMDENAS
from xnas.search_algorithm.MIGO import MIGO
from xnas.search_algorithm.DDPNAS import CategoricalDDPNAS


# Supported models
_spaces = {"darts": _DartsCNN, "nasbench201": _NASbench201, "ofa": build_super_net,
           "proxyless": build_super_net, "google": build_super_net, "pdarts": _PdartsCNN}

# Supported loss functions
_loss_funs = {"cross_entropy": torch.nn.CrossEntropyLoss}


def get_space():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.SPACE.NAME in _spaces.keys(), err_str.format(cfg.SPACE.NAME)
    return _spaces[cfg.SPACE.NAME]


def get_loss_fun():
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_space():
    """Builds the model."""
    return get_space()()


def build_loss_fun():
    """Build the loss function."""
    return get_loss_fun()()


def register_space(name, ctor):
    """Registers a model dynamically."""
    _spaces[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor


def sng_builder(category):
    if cfg.SNG.NAME == 'MDENAS':
        return CategoricalMDENAS(category, cfg.SNG.THETA_LR)
    elif cfg.SNG.NAME == 'DDPNAS':
        return CategoricalDDPNAS(category, cfg.SNG.PRUNING_STEP)
    elif cfg.SNG.NAME == 'SNG':
        return SNG(category)
    elif cfg.SNG.NAME == 'ASNG':
        return ASNG(category)
    elif cfg.SNG.NAME == 'dynamic_SNG':
        return Dynamic_SNG(category, step=cfg.SNG.PRUNING_STEP, pruning=cfg.SNG.PRUNING)
    elif cfg.SNG.NAME == 'dynamic_ASNG':
        return Dynamic_ASNG(category, step=cfg.SNG.PRUNING_STEP, pruning=cfg.SNG.PRUNING)
    elif cfg.SNG.NAME == 'MIGO':
        return MIGO(categories=category,
                    step=cfg.SNG.PRUNING_STEP,
                    pruning=cfg.SNG.PRUNING, sample_with_prob=cfg.SNG.PROB_SAMPLING,
                    utility_function=cfg.SNG.UTILITY, utility_function_hyper=cfg.SNG.UTILITY_FACTOR,
                    momentum=cfg.SNG.MOMENTUM, gamma=cfg.SNG.GAMMA)
    else:
        raise NotImplementedError


def lr_scheduler_builder(w_optim):
    if cfg.OPTIM.LR_POLICY == 'cos':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optim, cfg.OPTIM.MAX_EPOCH, eta_min=cfg.OPTIM.MIN_LR)
    elif cfg.OPTIM.LR_POLICY == 'step':
        return torch.optim.lr_scheduler.MultiStepLR(w_optim, cfg.OPTIM.STEPS, gamma=cfg.OPTIM.LR_MULT)
    else:
        raise NotImplementedError
