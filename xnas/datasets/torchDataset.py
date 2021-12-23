import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms


def getTorchDataset(name, root, cutout_length, batch_size, num_workers=8, download=True):
    assert name in ['cifar10', 'cifar100', 'svhn'], "dataset not support."
    if name == 'cifar10':
        train_transform, valid_transform = _transforms_cifar10(cutout_length)
        train_data = dset.CIFAR10(root=root, train=True, download=download, transform=train_transform)
        test_data = dset.CIFAR10(root=root, train=False, download=download, transform=valid_transform)
    elif name == 'cifar100':
        train_transform, valid_transform = _transforms_cifar100(cutout_length)
        train_data = dset.CIFAR100(root=root, train=True, download=download, transform=train_transform)
        test_data = dset.CIFAR100(root=root, train=True, download=download, transform=valid_transform)
    elif name == 'svhn':
        train_transform, valid_transform = _transforms_svhn(cutout_length)
        train_data = dset.SVHN(root=root, split='train', download=download, transform=train_transform)
        test_data = dset.SVHN(root=root, split='test', download=download, transform=train_transform)
    else:
        exit(0)

    train_loader = data.DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = data.DataLoader(
        dataset=test_data, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader


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


def _transforms_svhn(cutout_length):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(SVHN_MEAN, SVHN_STD),]
    )
    return train_transform, valid_transform


def _transforms_cifar100(cutout_length):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if cutout_length:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    )
    return train_transform, valid_transform


def _transforms_cifar10(cutout_length):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if cutout_length:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    )
    return train_transform, valid_transform

