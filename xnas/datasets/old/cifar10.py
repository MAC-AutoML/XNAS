# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR10 dataset."""

import os
import pickle

import numpy as np
from numpy.core.defchararray import index
from numpy.lib import index_tricks
import xnas.core.logging as logging
import xnas.datasets.transforms as transforms
import torch.utils.data
from xnas.core.config import cfg

import torchvision.transforms as torch_trans


logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [125.3, 123.0, 113.9]
_SD = [63.0, 62.1, 66.7]

# TODO: DALI backend support

def XNAS_Cifar10(data_path, split, backend='custom', batch_size=256, num_workers=4):
    """
        XNAS cifar10, generate dataloader from cifar10 train according split and beckend
        not support distributed now
    """
    if backend == 'custom':
        train_data = Cifar10(data_path, 'train')
        num_train = len(train_data)
        indices = list(range(num_train))
        # Shuffle data
        np.random.shuffle(indices)
        data_loaders = []
        pre_partition = 0.
        pre_index = 0
        for _split in split:
            current_partition = pre_partition + _split
            current_index = int(num_train * current_partition)
            current_indices = indices[pre_index: current_index]
            assert not len(current_indices) == 0, "Length of indices is zero!"
            _sampler = torch.utils.data.sampler.SubsetRandomSampler(
                current_indices)
            _data_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=batch_size,
                                                       sampler=_sampler,
                                                       num_workers=num_workers,
                                                       pin_memory=True
                                                       )
            data_loaders.append(_data_loader)
            pre_partition = current_partition
            pre_index = current_index
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
        inputs = inputs.reshape(
            (-1, 3, cfg.SEARCH.IM_SIZE, cfg.SEARCH.IM_SIZE))
        return inputs, labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = transforms.color_norm(im, _MEAN, _SD)
        if self._split == "train":
            im = transforms.horizontal_flip(im=im, p=0.5)
            im = transforms.random_crop(
                im=im, size=cfg.SEARCH.IM_SIZE, pad_size=4)
        return im

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = self._prepare_im(im)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]


def data_transforms_cifar10(cutout_length):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = torch_trans.Compose([
        torch_trans.RandomCrop(32, padding=4),
        torch_trans.RandomHorizontalFlip(),
        torch_trans.ToTensor(),
        torch_trans.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = torch_trans.Compose([
        torch_trans.ToTensor(),
        torch_trans.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img
