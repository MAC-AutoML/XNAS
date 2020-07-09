#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from xnas.core.config import cfg
from xnas.datasets.cifar10 import XNAS_Cifar10
from xnas.datasets.imagenet import XNAS_ImageFolder
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


# Supported datasets
_DATASETS = {"cifar10": XNAS_Cifar10, "imagenet": XNAS_ImageFolder}


def _construct_loader(dataset_name, split_list, batch_size):
    # Default data directory (/path/pycls/pycls/datasets/data)
    # _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    if cfg.DATA_LOADER.MEMORY_DATA:
        _DATA_DIR = "/userhome/temp_data"
    else:
        _DATA_DIR = "/gdata"
    # Relative data paths to default data directory
    _PATHS = {"cifar10": "cifar10/cifar-10-batches-py",
              "imagenet": "ImageNet2012"}
    """Constructs the data loader for the given dataset."""
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    print("reading data from {}".format(data_path))
    # Construct the dataset
    loader = _DATASETS[dataset_name](
        data_path, split_list,  batch_size)
    return loader


def shuffle(loader, cur_epoch):
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
