"""ImageNet dataset."""

import os
import re

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.datasets.transforms_imagenet import get_data_transform


logger = logging.get_logger(__name__)


class ImageFolder():
    """New ImageFolder
    Support ImageNet only currently.
    """
    def __init__(self, datapath, batch_size, split=None, use_val=False, augment_type='default', **kwargs):
        datapath = './data/imagenet/' if not datapath else datapath
        assert os.path.exists(datapath), "Data path '{}' not found".format(datapath)
        
        self.use_val = use_val
        self.data_path = datapath
        self.split = split
        if isinstance(batch_size, int):
            batch_size = [batch_size, batch_size]
        elif batch_size is None:
            batch_size = [256, 256]
        # assert len(batch_size) == len(split), "lengths of batch_size and split should be same."
        self.batch_size = batch_size
        self.num_workers = cfg.LOADER.NUM_WORKERS
        self.pin_memory = cfg.LOADER.PIN_MEMORY
        self.augment_type = augment_type
        self.kwargs = kwargs
        
        if not self.use_val:
            assert sum(self.split) == 1, "Summation of split should be 1."
        
        # setting default loader if not using MultiSizeRandomCrop
        if len(cfg.SEARCH.MULTI_SIZES) == 0:
            self.loader = torch.utils.data.DataLoader
        else:
            from xnas.datasets.utils.msrc_loader import msrc_DataLoader
            self.loader = msrc_DataLoader
            logger.info("Using MultiSize RandomCrop")

        # Acquiring transforms
        logger.info("Constructing transforms")
        self.train_transform, self.test_transform = self._build_transfroms()
        
        # Read all datasets
        logger.info("Constructing ImageFolder")
        self._construct_imdb()
    
    def _construct_imdb(self):
        # Images are stored per class in subdirs (format: n<number>)
        if not self.use_val:
            split_files = os.listdir(self.data_path)
        else:
            split_files = os.listdir(os.path.join(self.data_path, "train"))
        # imagenet format folder names
        self._class_ids = sorted(
            f for f in split_files if re.match(r"^n[0-9]+$", f))

        # Map class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        if not self.use_val:
            self._imdb = []
            for class_id in self._class_ids:
                cont_id = self._class_id_cont_id[class_id]
                train_im_dir = os.path.join(self.data_path, class_id)
                for im_name in os.listdir(train_im_dir):
                    im_path = os.path.join(train_im_dir, im_name)
                    if is_image_file(im_path):
                        self._imdb.append({"im_path": im_path, "class": cont_id})
            logger.info("Number of images: {}".format(len(self._imdb)))
            logger.info("Number of classes: {}".format(len(self._class_ids)))
        else:
            self._train_imdb = []
            self._val_imdb = []
            train_path = os.path.join(self.data_path, "train")
            val_path = os.path.join(self.data_path, "val")
            for class_id in self._class_ids:
                cont_id = self._class_id_cont_id[class_id]
                train_im_dir = os.path.join(train_path, class_id)
                for im_name in os.listdir(train_im_dir):
                    im_path = os.path.join(train_im_dir, im_name)
                    if is_image_file(im_path):
                        self._train_imdb.append({"im_path": im_path, "class": cont_id})
                val_im_dir = os.path.join(val_path, class_id)
                for im_name in os.listdir(val_im_dir):
                    im_path = os.path.join(val_im_dir, im_name)
                    if is_image_file(im_path):
                        self._val_imdb.append({"im_path": im_path, "class": cont_id})
            logger.info("Number of classes: {}".format(len(self._class_ids)))
            logger.info("Number of TRAIN images: {}".format(len(self._train_imdb)))
            logger.info("Number of VAL images: {}".format(len(self._val_imdb)))
    
    def _build_transfroms(self):
        # KWARGS for 'auto_augment_tf': policy='v0', interpolation='bilinear'
        return get_data_transform(augment=self.augment_type, **self.kwargs)
    
    def generate_data_loader(self):
        if not self.use_val:
            indices = list(range(len(self._imdb)))
            # Shuffle data
            np.random.shuffle(indices)
            data_loaders = []
            pre_partition = 0.
            pre_index = 0
            for i, _split in enumerate(self.split):
                _current_partition = pre_partition + _split
                _current_index = int(len(self._imdb) * _current_partition)
                _current_indices = indices[pre_index: _current_index]
                assert not len(_current_indices) == 0, "The length of indices is zero!"
                dataset = ImageList_torch(
                    [self._imdb[j] for j in _current_indices],
                    # using the first split only as training dataset
                    transform=self.train_transform if i==0 else self.test_transform
                )
                sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
                loader = self.loader(dataset,
                                    batch_size=self.batch_size[i],
                                    shuffle=(False if sampler else True),
                                    sampler=sampler,
                                    drop_last=(True if i==0 else False),
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory)
                data_loaders.append(loader)
                pre_partition = _current_partition
                pre_index = _current_index
            return data_loaders
        else:
            train_dataset = ImageList_torch(self._train_imdb, self.train_transform)
            train_sampler = DistributedSampler(train_dataset) if cfg.NUM_GPUS > 1 else None
            train_loader = self.loader(train_dataset,
                                        batch_size=self.batch_size[0],
                                        shuffle=(False if train_sampler else True),
                                        sampler=train_sampler,
                                        drop_last=True,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory)
            
            val_dataset = ImageList_torch(self._val_imdb, self.test_transform)
            val_sampler = DistributedSampler(val_dataset) if cfg.NUM_GPUS > 1 else None
            valid_loader = self.loader(val_dataset,
                                        batch_size=self.batch_size[1],
                                        shuffle=(False if val_sampler else True),
                                        sampler=val_sampler,
                                        drop_last=False,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory)
            return [train_loader, valid_loader]

class ImageList_torch(torch.utils.data.Dataset):
    '''
        ImageList dataloader with torch backends
        From https://github.com/pytorch/vision/issues/81
    '''

    def __init__(self, list, transform):
        self._imdb = list
        self.transform = transform
        self.loader = pil_loader

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
