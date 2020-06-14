from datasets import get_data
import torch
import numpy as np
import tqdm

data_dict = [
    {'name': 'cifar10',
     'data_path': '/userhome/temp_data/cifar10'},
    {'name': 'cifar100',
     'data_path': '/userhome/temp_data/cifar100'},
    {'name': 'fashionmnist',
     'data_path': '/userhome/temp_data/fashionmnist'},
]

for data in data_dict:
    input_size, input_channels, n_classes, train_data = get_data.get_data(
        data['name'], data['data_path'], cutout_length=0, validation=False,
        image_size=None)
    n_train = len(train_data)
    split = n_train - int(n_train / 10.)
    indices = list(range(n_train))
    # shuffle data
    np.random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=256,
                                               sampler=train_sampler,
                                               num_workers=2,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=256,
                                               sampler=valid_sampler,
                                               num_workers=2,
                                               pin_memory=True)
    print("Test searching dataloader")
    for (trn_X, trn_y) in tqdm.tqdm(train_loader):
        # print(trn_X.shape)
        # print(trn_y.shape)
        pass
    print("Test validation dataloader")
    for (trn_X, trn_y) in tqdm.tqdm(valid_loader):
        # print(trn_X.shape)
        # print(trn_y.shape)
        pass
