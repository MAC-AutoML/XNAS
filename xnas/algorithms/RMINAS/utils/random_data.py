import torch
import random
import numpy as np
from xnas.datasets.loader import get_normal_dataloader
from xnas.datasets.imagenet import ImageFolder


def get_random_data(batchsize, name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if name == 'imagenet':
        train_loader, _ = ImageFolder(
            datapath="./data/imagenet/ILSVRC2012_img_train/", 
            batch_size=batchsize*16,
            split=[0.5, 0.5], 
        ).generate_data_loader()
    else:
        train_loader, _ = get_normal_dataloader(name, batchsize*16)
    
    random_idxs = np.random.randint(0, len(train_loader.dataset), size=train_loader.batch_size)
    (more_data_X, more_data_y) = zip(*[train_loader.dataset[idx] for idx in random_idxs])
    more_data_X = torch.stack(more_data_X, dim=0).to(device)
    more_data_y = torch.Tensor(more_data_y).long().to(device)
    return more_data_X, more_data_y
