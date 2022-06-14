"""Image transformations."""

import time
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


__all__ = [
    "transforms_svhn", 
    "transforms_cifar100", 
    "transforms_cifar10", 
    "transforms_imagenet16",
    "transforms_mnist",
    "transforms_fashionmnist",
]


def transforms_mnist(cutout_length):
    MEAN = [0.13066051707548254]
    STD = [0.30810780244715075]
    
    train_transform = transforms.Compose(
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    )
    if cutout_length:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    )
    return train_transform, valid_transform


def transforms_fashionmnist(cutout_length):
    MEAN = [0.28604063146254594]
    STD = [0.35302426207299326]
    
    train_transform = transforms.Compose(
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    )
    if cutout_length:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    )
    return train_transform, valid_transform


def transforms_svhn(cutout_length):
    MEAN = [0.4377, 0.4438, 0.4728]
    STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    if cutout_length:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, valid_transform


def transforms_cifar100(cutout_length):
    MEAN = [0.5071, 0.4865, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    if cutout_length:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, valid_transform


def transforms_cifar10(cutout_length):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    if cutout_length:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, valid_transform


def transforms_imagenet16():
    MEAN = [0.48109804, 0.45749020, 0.40788235]
    STD = [0.24792157, 0.24023529, 0.25525490]

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    # Cutout is not used here.

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
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

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class MultiSizeRandomCrop(transforms.RandomResizedCrop):
    """Random multi-sized crop"""
    
    ACTIVE_SIZE = 224
    CANDIDATE_SIZES = [224]
    
    def __init__(
        self,
        size_list,
        continuous=True,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
    ):
        self.IMAGE_SIZE_SEG = 4
        self.SIZE_LIST = size_list
        self.CONTINUOUS = continuous
        MultiSizeRandomCrop.ACTIVE_SIZE = max(self.SIZE_LIST)
        super(MultiSizeRandomCrop, self).__init__(MultiSizeRandomCrop.ACTIVE_SIZE, scale, ratio)
        MultiSizeRandomCrop.CANDIDATE_SIZES, _ = self.get_candidate_image_size()   # do not use weighted random sampling
        self.sample_image_size()

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(
            img, i, j, h, w,
            (MultiSizeRandomCrop.ACTIVE_SIZE, MultiSizeRandomCrop.ACTIVE_SIZE),
        )
    
    def get_candidate_image_size(self):
        if self.CONTINUOUS:
            min_size = min(self.SIZE_LIST)
            max_size = max(self.SIZE_LIST)
            candidate_sizes = []
            for i in range(min_size, max_size + 1):
                if i % self.IMAGE_SIZE_SEG == 0:
                    candidate_sizes.append(i)
        else:
            candidate_sizes = self.SIZE_LIST

        relative_probs = None   # weighted random choices
        return candidate_sizes, relative_probs

    @staticmethod
    def sample_image_size():
        _seed = time.time()
        random.seed(_seed)
        MultiSizeRandomCrop.ACTIVE_SIZE = random.choices(MultiSizeRandomCrop.CANDIDATE_SIZES)[0]
