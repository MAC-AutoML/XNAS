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
from xnas.search_space.cell_based import _PdartsCNN


# Supported models
_spaces = {"darts": _DartsCNN, "nasbench201": _NASbench201, "pdarts": _PdartsCNN}

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
