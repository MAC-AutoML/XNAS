import os
import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageFolder


def cifar10_data(batchsize, workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    _train_loader = torch.utils.data.DataLoader(
        CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batchsize*16, shuffle=True,
        num_workers=workers, pin_memory=True)

    target_i = random.randint(0, len(_train_loader)-1)
    more_data_X, more_data_y = None, None
    for i, (more_data_X, more_data_y) in enumerate(_train_loader):
        if i == target_i:
            break
    more_data_X = more_data_X.cuda()
    more_data_y = more_data_y.cuda()
    return more_data_X, more_data_y


def cifar100_data(batchsize, workers):
    CIFAR100_TRAIN_MEAN = (
        0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (
        0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    cifar100_training = CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = torch.utils.data.DataLoader(
        cifar100_training, shuffle=True, 
        batch_size=batchsize*16, num_workers=workers)

    target_i = random.randint(0, len(cifar100_training_loader)-1)
    more_data_X, more_data_y = None, None
    for i, (more_data_X, more_data_y) in enumerate(cifar100_training_loader):
        if i == target_i:
            break
    more_data_X = more_data_X.cuda()
    more_data_y = more_data_y.cuda()
    return more_data_X, more_data_y

def imagenet_data(batchsize, workers, data_dir='/gdata/ImageNet2012/'):
    """Data preparing"""
    traindir = os.path.join(data_dir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batchsize*16, shuffle=True, pin_memory=True, num_workers=workers)

    target_i = random.randint(0, len(train_loader)-1)
    more_data_X, more_data_y = None, None
    for i, (more_data_X, more_data_y) in enumerate(train_loader):
        if i == target_i:
            break
    more_data_X = more_data_X.cuda()
    more_data_y = more_data_y.cuda()
    return more_data_X, more_data_y