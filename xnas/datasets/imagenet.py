# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import math
import os
import re

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

import xnas.logger.logging as logging
from xnas.core.config import cfg


logger = logging.get_logger(__name__)


class ImageFolder():
    def __init__(
            self,
            datapath,
            split,
            batch_size=None,
            dataset_name='imagenet',
            _rgb_normalized_mean=None,
            _rgb_normalized_std=None,
            transforms=None,
            num_workers=cfg.LOADER.NUM_WORKERS,
            pin_memory=cfg.LOADER.PIN_MEMORY,
            shuffle=True
        ):
        assert os.path.exists(datapath), "Data path '{}' not found".format(datapath)
        assert sum(split) == 1, "Summation of split should be 1"
        
        self._data_path, self._split, self.dataset_name = datapath, split, dataset_name
        self._rgb_normalized_mean, self._rgb_normalized_std = _rgb_normalized_mean, _rgb_normalized_std
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        if transforms is None:
            self.transforms = [{'crop': 'random', 'crop_size': cfg.SEARCH.IM_SIZE, 'min_crop': 0.08, 'random_flip': True},
                               {'crop': 'center', 'crop_size': cfg.TEST.IM_SIZE, 'min_crop': -1, 'random_flip': False}]  # NOTE: min_crop is not used here.
        else:
            self.transforms = transforms
        assert len(self.transforms) == len(self._split), "Length of split and transforms should be equal"
        
        # Read all dataset
        logger.info("Constructing ImageFolder")
        self._construct_imdb()

    def _construct_imdb(self):
        # Images are stored per class in subdirs (format: n<number>)
        split_files = os.listdir(self._data_path)
        if self.dataset_name == "imagenet":
            # imagenet format folder names
            self._class_ids = sorted(
                f for f in split_files if re.match(r"^n[0-9]+$", f))
            self._rgb_normalized_mean = [0.485, 0.456, 0.406]
            self._rgb_normalized_std = [0.229, 0.224, 0.225]
        elif self.dataset_name == 'custom':
            self._class_ids = sorted(
                f for f in split_files if not f[0] == '.')
        else:
            raise NotImplementedError

        # Map class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(self._data_path, class_id)
            for im_name in os.listdir(im_dir):
                im_path = os.path.join(im_dir, im_name)
                if is_image_file(im_path):
                    self._imdb.append({"im_path": im_path, "class": cont_id})
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))
        # return self._imdb

    def generate_data_loader(self):
        indices = list(range(len(self._imdb)))
        # Shuffle data
        np.random.shuffle(indices)
        data_loaders = []
        pre_partition = 0.
        pre_index = 0
        for i, _split in enumerate(self._split):
            _current_partition = pre_partition + _split
            _current_index = int(len(self._imdb) * _current_partition)
            _current_indices = indices[pre_index: _current_index]
            assert not len(
                _current_indices) == 0, "The length of indices is zero!"
            dataset = ImageList_torch([self._imdb[j] for j in _current_indices],
                                       _rgb_normalized_mean=self._rgb_normalized_mean,
                                       _rgb_normalized_std=self._rgb_normalized_std, **self.transforms[i])
            sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
            loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size[i],
                                                 shuffle=(False if sampler else True),
                                                 sampler=sampler,
                                                 num_workers=self.num_workers,
                                                 pin_memory=self.pin_memory)
            data_loaders.append(loader)
            pre_partition = _current_partition
            pre_index = _current_index
        return data_loaders


class ImageList_torch(torch.utils.data.Dataset):
    '''
        ImageList dataloader with torch backends
        From https://github.com/pytorch/vision/issues/81
    '''

    def __init__(
            self,
            _list,
            _rgb_normalized_mean=None,
            _rgb_normalized_std=None,
            crop='random',
            crop_size=224,
            min_crop=0.08,
            random_flip=False):
        self._imdb = _list
        self._bgr_normalized_mean = _rgb_normalized_mean[::-1]
        self._bgr_normalized_std = _rgb_normalized_std[::-1]
        self.crop = crop
        self.crop_size = crop_size
        self.min_crop = min_crop
        self.random_flip = random_flip
        self.loader = pil_loader
        self._construct_transforms()

    def _construct_transforms(self):
        transforms = []
        if self.crop == "random":
            transforms.append(torch_transforms.RandomResizedCrop(self.crop_size, scale=(self.min_crop, 1.0)))
        elif self.crop == "center":
            transforms.append(torch_transforms.Resize(math.ceil(self.crop_size / 0.875)))   # assert crop_size==224
            transforms.append(torch_transforms.CenterCrop(self.crop_size))
        # TODO: color augmentation support
        # transforms.append(torch_transforms.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4, hue=0.2))
        if self.random_flip:
            transforms.append(torch_transforms.RandomHorizontalFlip())
        transforms.append(torch_transforms.ToTensor())
        transforms.append(torch_transforms.Normalize(mean=self._bgr_normalized_mean, std=self._bgr_normalized_std))
        self.transform = torch_transforms.Compose(transforms)

    def __getitem__(self, index):
        impath = self._imdb[index]["im_path"]
        target = self._imdb[index]["class"]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self._imdb)


def is_image_file(filename):
    IMG_EXTENSIONS = (".jpg", ".jpeg",".png",".ppm",".bmp",".pgm",".tif",".tiff",".webp",)
    return filename.lower().endswith(IMG_EXTENSIONS)


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
