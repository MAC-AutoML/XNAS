import xnas.core.config as config
from xnas.datasets.loader import construct_loader
from xnas.core.config import cfg

config.load_configs()

# cifar10
[train_loader, valid_loader] = construct_loader()

# cifar100
cfg.LOADER.DATASET = 'cifar100'
cfg.LOADER.NUM_CLASSES = 100
[train_loader, valid_loader] = construct_loader()

# imagenet16
cfg.LOADER.DATASET = 'imagenet16'
cfg.LOADER.NUM_CLASSES = 120
[train_loader, valid_loader] = construct_loader()

