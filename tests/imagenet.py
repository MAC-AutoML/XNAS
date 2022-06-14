import xnas.core.config as config
from xnas.datasets.loader import construct_loader
from xnas.core.config import cfg

config.load_configs()

[train_loader, valid_loader] = construct_loader()

for i, (trn_X, trn_y) in enumerate(train_loader):
    print(trn_X.shape, trn_y.shape)
    if i==9:
        break

# cfg.SEARCH.MULTI_SIZES = []

print("===")
for i, (trn_X, trn_y) in enumerate(valid_loader):
    print(trn_X.shape, trn_y.shape)
    if i==9:
        break
