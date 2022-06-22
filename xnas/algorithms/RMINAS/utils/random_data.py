import torch
import random
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
    
    target_i = random.randint(0, len(train_loader)-1)
    more_data_X, more_data_y = None, None
    for i, (more_data_X, more_data_y) in enumerate(train_loader):
        if i == target_i:
            break
    more_data_X = more_data_X.to(device)
    more_data_y = more_data_y.to(device)
    return more_data_X, more_data_y
