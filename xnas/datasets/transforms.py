"""Image transformations."""

import numpy as np
import torch
import torchvision.transforms as transforms


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
