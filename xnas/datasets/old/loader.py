# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

from xnas.core.config import cfg
from xnas.datasets.old.cifar10 import XNAS_Cifar10
from xnas.datasets.imagenet import XNAS_ImageFolder
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler


# Supported datasets
_DATASETS = {"cifar10": XNAS_Cifar10, "imagenet": XNAS_ImageFolder}
# Relative data paths to default data directory
_PATHS = {"cifar10": "cifar-10-batches-py",
          "imagenet": "ImageNet2012"}

def construct_loader(dataset_name, split_list, batch_size, datapath=None):
    # Default data directory (/path/pycls/pycls/datasets/data)
    if cfg.DATA_LOADER.MEMORY_DATA:
        _DATA_DIR = "/userhome/temp_data"
    elif datapath is not None:
        _DATA_DIR = datapath
    else:
        _DATA_DIR = "/gdata"
    # Constructs the data loader for the given dataset
    assert dataset_name in _DATASETS and dataset_name in _PATHS, "Dataset '{}' not supported".format(
        dataset_name)
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    print("reading data from {}".format(data_path))
    # Construct the dataset
    loader = _DATASETS[dataset_name](
        data_path, split_list, backend=cfg.DATA_LOADER.BACKEND, batch_size=batch_size, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
    return loader


# def shuffle(loader, cur_epoch):
#     err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
#     assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
#     # RandomSampler handles shuffling automatically
#     if isinstance(loader.sampler, DistributedSampler):
#         # DistributedSampler shuffles data based on epoch
#         loader.sampler.set_epoch(cur_epoch)
