import numpy as np
import torch.utils.data as data
import torchvision.datasets as dset

from xnas.datasets.transforms import *
from xnas.datasets.imagenet16 import ImageNet16
from xnas.datasets.imagenet import XNAS_ImageFolder

SUPPORT_DATASETS = ["cifar10", "cifar100", "svhn", "imagenet16"]
# if you use datasets loaded by imagefolder, you can add it here.
IMAGEFOLDER_FORMAT = ["imagenet"]


def construct_loader(
    name,
    split,
    batch_size,
    datapath=None,
    cutout_length=0,
    num_workers=8,
    use_classes=None,
    backend="torch",
):
    assert (name in SUPPORT_DATASETS) or (
        name in IMAGEFOLDER_FORMAT
    ), "dataset not supported."
    datapath = "./data/" + name if datapath is None else datapath

    if name in SUPPORT_DATASETS:
        train_data, _ = getData(name, datapath, cutout_length, use_classes)
        return splitDataLoader(train_data, batch_size, split, num_workers)
    else:
        data_ = XNAS_ImageFolder(
            datapath, split, backend, batch_size=batch_size, num_workers=num_workers
        )
        return data_.generate_data_loader()


def getData(name, root, cutout_length, download=True, use_classes=None):
    assert name in SUPPORT_DATASETS, "dataset not support."
    assert cutout_length >= 0, "cutout_length should not be less than zero."

    if name == "cifar10":
        train_transform, valid_transform = transforms_cifar10(cutout_length)
        train_data = dset.CIFAR10(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "cifar100":
        train_transform, valid_transform = transforms_cifar100(cutout_length)
        train_data = dset.CIFAR100(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=root, train=True, download=download, transform=valid_transform
        )
    elif name == "svhn":
        train_transform, valid_transform = transforms_svhn(cutout_length)
        train_data = dset.SVHN(
            root=root, split="train", download=download, transform=train_transform
        )
        test_data = dset.SVHN(
            root=root, split="test", download=download, transform=valid_transform
        )
    elif name == "imagenet16":
        train_transform, valid_transform = transforms_imagenet16()
        train_data = ImageNet16(
            root=root,
            train=True,
            transform=train_transform,
            use_num_of_class_only=use_classes,
        )
        test_data = ImageNet16(
            root=root,
            train=False,
            transform=valid_transform,
            use_num_of_class_only=use_classes,
        )
        if use_classes == 120:
            assert len(train_data) == 151700
    else:
        exit(0)
    return train_data, test_data


def getDataLoader(
    name,
    root,
    batch_size,
    cutout_length,
    num_workers=8,
    download=True,
    use_classes=None,
):
    train_data, test_data = getData(name, root, cutout_length, download, use_classes)
    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def splitDataLoader(data_, batch_size, split, num_workers=8):
    assert 0 not in split, "illegal split list with zero."
    assert sum(split) == 1, "summation of split should be one."
    num_data = len(data_)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    portion = [int(sum(split[:i]) * num_data) for i in range(len(split) + 1)]

    return [
        data.DataLoader(
            dataset=data_,
            batch_size=batch_size,
            sampler=data.sampler.SubsetRandomSampler(
                indices[portion[i - 1] : portion[i]]
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        for i in range(1, len(portion))
    ]
