#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR10 dataset."""

import os
import pickle

import numpy as np
import xnas.core.logging as logging
import xnas.datasets.transforms as transforms
import torch.utils.data
from xnas.core.config import cfg


logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [125.3, 123.0, 113.9]
_SD = [63.0, 62.1, 66.7]

'''Xnas cifar10, generate dataloader from cifar10 train according split and beckend, do not support distributed now'''


def XNAS_Cifar10(data_path, split, backend='custom', batch_size=256, works=4):
    assert backend == 'custom'
    if backend == 'custom':
        train_data = Cifar10(data_path, 'train')
        n_train = len(train_data)
        indices = list(range(n_train))
        # shuffle data
        np.random.shuffle(indices)
        data_loaders = []
        pre_partition = 0.
        pre_index = 0
        for i, _split in enumerate(split):
            _current_partition = pre_partition + _split
            _current_index = int(len(train_data) * _current_partition)
            _current_indices = indices[pre_index: _current_index]
            assert not len(
                _current_indices) == 0, "The length of indices is zero!"
            _sampler = torch.utils.data.sampler.SubsetRandomSampler(
                _current_indices)
            _data_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=batch_size,
                                                       sampler=_sampler,
                                                       num_workers=works,
                                                       pin_memory=True)
            data_loaders.append(_data_loader)
            pre_partition = _current_partition
            pre_index = _current_index
        return data_loaders
    else:
        raise NotImplementedError


class Cifar10(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for cifar".format(
            split)
        logger.info("Constructing CIFAR-10 {}...".format(split))
        self._data_path, self._split = data_path, split
        self._inputs, self._labels = self._load_data()

    def _load_data(self):
        """Loads data into memory."""
        logger.info("{} data path: {}".format(self._split, self._data_path))
        # Compute data batch names
        if self._split == "train":
            batch_names = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            batch_names = ["test_batch"]
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._data_path, batch_name)
            with open(batch_path, "rb") as f:
                data = pickle.load(f, encoding="bytes")
            inputs.append(data[b"data"])
            labels += data[b"labels"]
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)
        inputs = inputs.reshape((-1, 3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE))
        return inputs, labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = transforms.color_norm(im, _MEAN, _SD)
        if self._split == "train":
            im = transforms.horizontal_flip(im=im, p=0.5)
            im = transforms.random_crop(
                im=im, size=cfg.TRAIN.IM_SIZE, pad_size=4)
        return im

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = self._prepare_im(im)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]
