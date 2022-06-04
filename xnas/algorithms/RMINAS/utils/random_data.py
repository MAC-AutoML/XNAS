import random
from xnas.datasets.loader import get_normal_dataloader
from xnas.datasets.imagenet import ImageFolder


def get_random_data(batchsize, name):
    if name == 'imagenet':
        train_loader, _ = ImageFolder(
            "./data/imagenet/ILSVRC2012_img_train/", 
            [0.5, 0.5], 
            batchsize*16,
        ).generate_data_loader()
    else:
        train_loader, _ = get_normal_dataloader(name, batchsize*16)
    
    target_i = random.randint(0, len(train_loader)-1)
    more_data_X, more_data_y = None, None
    for i, (more_data_X, more_data_y) in enumerate(train_loader):
        if i == target_i:
            break
    more_data_X = more_data_X.cuda()
    more_data_y = more_data_y.cuda()
    return more_data_X, more_data_y