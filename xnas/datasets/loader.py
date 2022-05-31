import os
import numpy as np
import torch.utils.data as data
import torchvision.datasets as dset

from xnas.core.config import cfg
from xnas.datasets.transforms import *
from xnas.datasets.imagenet16 import ImageNet16
from xnas.datasets.imagenet import ImageFolder


SUPPORTED_DATASETS = [
    "cifar10", 
    "cifar100", 
    "svhn", 
    "imagenet16", 
    "mnist", 
    "fashionmnist"
]

# if you use datasets loaded by imagefolder, you can add it here.
IMAGEFOLDER_FORMAT = ["imagenet"]


def construct_loader(
        cutout_length=0,
        use_classes=None,
        transforms=None,
    ):
    """Construct NAS dataloaders with train&valid subsets."""
    
    split = cfg.LOADER.SPLIT
    name = cfg.LOADER.DATASET
    batch_size = cfg.LOADER.BATCH_SIZE 
    datapath = cfg.LOADER.DATAPATH
    
    assert (name in SUPPORTED_DATASETS) or (name in IMAGEFOLDER_FORMAT), "dataset not supported."

    # expand batch_size to support different number during training & validating
    if isinstance(batch_size, int):
        batch_size = [batch_size, batch_size]
    elif batch_size is None:
        batch_size = [256, 256]
    assert len(batch_size) == len(split), "lengths of batch_size and split should be same."
    
    # check if randomresized crop is used only in ImageFolder type datasets
    if isinstance(cfg.SEARCH.IM_SIZE, list):
        assert name in IMAGEFOLDER_FORMAT, "RandomResizedCrop can only be used in ImageFolder currently."
    
    if name in SUPPORTED_DATASETS:
        # using training data only.
        train_data, _ = get_data(name, datapath, cutout_length, use_classes=use_classes, transforms=transforms)
        return split_dataloader(train_data, batch_size, split)
    elif name in IMAGEFOLDER_FORMAT:
        return ImageFolder( # using path of training data of ImageNet as `datapath`
            datapath, split, batch_size=batch_size,
            transforms=transforms,
        ).generate_data_loader()
    else:
        print("dataset not supported.")
        exit(0) 


def get_data(name, root, cutout_length, download=True, use_classes=None, transforms=None):
    assert name in SUPPORTED_DATASETS, "dataset not support."
    assert cutout_length >= 0, "cutout_length should not be less than zero."
    root = "./data/" + name if not root else os.path.join(root, name)

    if name == "cifar10":
        train_transform, valid_transform = transforms_cifar10(cutout_length) if transforms is None else transforms
        train_data = dset.CIFAR10(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "cifar100":
        train_transform, valid_transform = transforms_cifar100(cutout_length) if transforms is None else transforms
        train_data = dset.CIFAR100(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "svhn":
        train_transform, valid_transform = transforms_svhn(cutout_length) if transforms is None else transforms
        train_data = dset.SVHN(
            root=root, split="train", download=download, transform=train_transform
        )
        test_data = dset.SVHN(
            root=root, split="test", download=download, transform=valid_transform
        )
    elif name == "mnist":
        train_transform, valid_transform = transforms_mnist(cutout_length) if transforms is None else transforms
        train_data = dset.MNIST(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.MNIST(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "fashionmnist":
        train_transform, valid_transform = transforms_mnist(cutout_length) if transforms is None else transforms
        train_data = dset.FashionMNIST(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.FashionMNIST(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "imagenet16":
        train_transform, valid_transform = transforms_imagenet16() if transforms is None else transforms
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
        if use_classes == 120 or use_classes is None:   # Use 120 classes by default.
            assert len(train_data) == 151700 and len(test_data) == 6000
        elif use_classes == 150:
            assert len(train_data) == 190272 and len(test_data) == 7500
        elif use_classes == 200:
            assert len(train_data) == 254775 and len(test_data) == 10000
        elif use_classes == 1000:
            assert len(train_data) == 1281167 and len(test_data) == 50000
    else:
        exit(0)
    return train_data, test_data


def get_normal_dataloader(
    name=None,
    train_batch=None,
    cutout_length=0,
    download=True,
    use_classes=None,
    transforms=None,
):
    name=cfg.LOADER.DATASET if name is None else name
    train_batch=cfg.LOADER.BATCH_SIZE if train_batch is None else train_batch
    name=cfg.LOADER.DATASET
    root=cfg.LOADER.DATAPATH
    test_batch=cfg.TEST.BATCH_SIZE
    
    # get normal dataloaders with train&test subsets.
    train_data, test_data = get_data(name, root, cutout_length, download, use_classes, transforms)
    
    # if loader.batch_size is a list for [train, val_1, ...], the first value will be used.
    if isinstance(train_batch, list):
        train_batch = train_batch[0]
        
    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=train_batch,
        shuffle=True,
        num_workers=cfg.LOADER.NUM_WORKERS,
        pin_memory=cfg.LOADER.PIN_MEMORY,
    )
    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=test_batch,
        shuffle=False,
        num_workers=cfg.LOADER.NUM_WORKERS,
        pin_memory=cfg.LOADER.PIN_MEMORY,
    )
    return train_loader, test_loader


def split_dataloader(data_, batch_size, split):
    assert 0 not in split, "illegal split list with zero."
    assert sum(split) == 1, "summation of split should be one."
    num_data = len(data_)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    portion = [int(sum(split[:i]) * num_data) for i in range(len(split) + 1)]

    return [
        data.DataLoader(
            dataset=data_,
            batch_size=batch_size[i - 1],
            sampler=data.sampler.SubsetRandomSampler(
                indices[portion[i - 1] : portion[i]]
            ),
            num_workers=cfg.LOADER.NUM_WORKERS,
            pin_memory=cfg.LOADER.PIN_MEMORY,
        )
        for i in range(1, len(portion))
    ]
