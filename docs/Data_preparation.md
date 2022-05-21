## Data Preparation

It is highly recommended to save or link datasets to the `$XNAS/data` folder, thus no additional configuration is required. 

However, manually setting the path for datasets is also available by modifying the `cfg.LOADER.DATAPATH` attribute in the configuration file `$XNAS/xnas/core/config.py`.

Additionally, files required by benchmarks are also in the `$XNAS/data` folder. You can also modify related attributes under `cfg.BENCHMARK` in the configuration file, to match your actual file locations.


### Supported Datasets

The dataloaders of XNAS will read the dataset files from `$XNAS/data/$DATASET_NAME` by default, and we use lowercase filenames and remove the hyphens. For example, files for CIFAR-10 should be placed (or auto downloaded) under `$XNAS/data/cifar/` directory.

XNAS currently supports the following datasets.

- CIFAR-10
- CIFAR-100
- ImageNet
  - ImageNet16 (Downsampled)
- SVHN
- MNIST
  - FashionMNIST

