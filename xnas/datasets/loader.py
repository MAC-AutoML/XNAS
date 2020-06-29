#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from xnas.core.config import cfg
from xnas.datasets.cifar10 import Cifar10
from xnas.datasets.imagenet import ImageNet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


# Supported datasets
_DATASETS = {"cifar10": Cifar10, "imagenet": ImageNet}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    # Default data directory (/path/pycls/pycls/datasets/data)
    # _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    if cfg.DATA_LOADER.MEMORY_DATA:
        _DATA_DIR = "/userhome/temp_data"
    else:
        _DATA_DIR = "/gdata"
    # Relative data paths to default data directory
    _PATHS = {"cifar10": "cifar10", "imagenet": "ImageNet2012"}
    """Constructs the data loader for the given dataset."""
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    print("reading data from {}".format(data_path))
    # Construct the dataset
    loader = _DATASETS[dataset_name](
        data_path, split,  batch_size, shuffle, drop_last)
    if not hasattr(loader, 'sampler'):
        setattr(loader, 'sampler', None)
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True,
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
