# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import gc
import importlib
from logging import currentframe
import math
import os
import re

import cv2
import numpy as np
from numpy.testing._private.utils import assert_equal
import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as torch_transforms
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from xnas.core import utils

import xnas.core.logging as logging
import xnas.datasets.transforms as custom_transforms
from xnas.core.config import cfg
from xnas.core.utils import random_time_string
from xnas.datasets.utils import default_loader, is_image_file

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    print('Could not import DALI')

logger = logging.get_logger(__name__)


class XNAS_ImageFolder():
    def __init__(
            self,
            data_path,
            split,
            backend,
            batch_size=None,
            dataset_name='imagenet',
            _rgb_normalized_mean=None,
            _rgb_normalized_std=None,
            transformers=None,
            num_workers=8,
            pin_memory=True,
            world_size=1,
            shuffle=True):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        assert sum(split) == 1, "Summation of split should be 1"
        assert backend in ['torch', 'custom', 'dali_cpu',
                           'dali_gpu'], "Corresponding backend {} is not supported!".format(backend)
        self._data_path, self._split, self.backend, self.dataset_name = data_path, split, backend, dataset_name
        self._rgb_normalized_mean, self._rgb_normalized_std = _rgb_normalized_mean, _rgb_normalized_std
        self.batch_size = [256, 200] if batch_size is None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.world_size = world_size
        self.shuffle = shuffle
        if transformers is None:
            self.transformers = [{'crop': 'random', 'crop_size': 224, 'min_crop_size': 0.08, 'random_flip': True},
                                 {'crop': 'center', 'crop_size': 224, 'min_crop_size': 256, 'random_flip': False}]
        else:
            self.transformers = transformers
        assert len(self.transformers) == len(
            self._split), "Length of split and transformers should be equal"
        # Read all dataset
        logger.info("Constructing XNAS_ImageFolder")
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
            if self._rgb_normalized_mean is None:
                logger.warning("image mean is None using imagenet mean value!")
                self._rgb_normalized_mean = [0.485, 0.456, 0.406]
            if self._rgb_normalized_std is None:
                logger.warning("image std is None using imagenet std value!")
                self._rgb_normalized_std = [0.229, 0.224, 0.225]
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
        print("Number of images: {}".format(len(self._imdb)))
        print("Number of classes: {}".format(len(self._class_ids)))

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
            if self.backend in ['custom', 'torch']:
                if self.backend == 'custom':
                    dataset = ImageList_custom([self._imdb[j] for j in _current_indices],
                                               _rgb_normalized_mean=self._rgb_normalized_mean,
                                               _rgb_normalized_std=self._rgb_normalized_std, **self.transformers[i])
                else:
                    dataset = ImageList_torch([self._imdb[j] for j in _current_indices],
                                              _rgb_normalized_mean=self._rgb_normalized_mean,
                                              _rgb_normalized_std=self._rgb_normalized_std, **self.transformers[i])
                sampler = DistributedSampler(
                    dataset) if cfg.NUM_GPUS > 1 else None
                loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size[i],
                                                     shuffle=(
                                                         False if sampler else True),
                                                     sampler=sampler,
                                                     num_workers=self.num_workers,
                                                     pin_memory=self.pin_memory)
            elif self.backend in ['dali_cpu', 'dali_gpu']:
                dali_cpu = True if self.backend == 'dali_cpu' else False
                loader = ImageList_DALI([self._imdb[j] for j in _current_indices], self.batch_size[i],
                                        _rgb_normalized_mean=self._rgb_normalized_mean,
                                        _rgb_normalized_std=self._rgb_normalized_std,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory,
                                        world_size=self.world_size,
                                        dali_cpu=dali_cpu,
                                        temp_folder='/tmp',
                                        **self.transformers[i])
                loader = loader.get_data_iter()
            data_loaders.append(loader)
            pre_partition = _current_partition
            pre_index = _current_index
        return data_loaders


class ImageList_custom(torch.utils.data.Dataset):
    """ImageNet dataset, custom backend."""

    def __init__(
            self,
            _list,
            _rgb_normalized_mean=None,
            _rgb_normalized_std=None,
            crop='random',
            crop_size=224,
            min_crop_size=0.08,
            random_flip=False):
        logger.info("Using Custom (opencv2 and numpy array) as backend.")
        self._imdb = _list
        self._bgr_normalized_mean = _rgb_normalized_mean[::-1]
        self._bgr_normalized_std = _rgb_normalized_std[::-1]
        self.crop = crop
        self.crop_size = crop_size
        self.min_crop_size = min_crop_size
        self.random_flip = random_flip

    def _prepare_im(self, im):
        """Prepare the image for network input."""
        # Train and test setups differ
        if self.crop == "random":
            # Scale -> aspect ratio -> crop
            im = custom_transforms.random_sized_crop(
                im=im, size=self.crop_size, area_frac=self.min_crop_size)
        elif self.crop == "center":
            # Scale -> center crop
            im = custom_transforms.scale(self.min_crop_size, im)
            im = custom_transforms.center_crop(self.crop_size, im)
        if self.random_flip:
            im = custom_transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        # HWC -> CHW
        im = im.transpose([2, 0, 1])
        # Normalize [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = custom_transforms.color_norm(
            im, self._bgr_normalized_mean, self._bgr_normalized_std)
        return im

    def __getitem__(self, index):
        # Load image
        im = cv2.imread(self._imdb[index]["im_path"])
        im = im.astype(np.float32, copy=False)
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)


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
            min_crop_size=0.08,
            random_flip=False):
        logger.info("Using Torch (PIL and torchvison transformer) as backend.")
        self._imdb = _list
        self._bgr_normalized_mean = _rgb_normalized_mean[::-1]
        self._bgr_normalized_std = _rgb_normalized_std[::-1]
        self.crop = crop
        self.crop_size = crop_size
        self.min_crop_size = min_crop_size
        self.random_flip = random_flip
        self.loader = default_loader
        self._construct_transformer()

    def _construct_transformer(self):
        transformer = []
        if self.crop == "random":
            transformer.append(torch_transforms.RandomResizedCrop(
                self.crop_size, scale=(self.min_crop_size, 1.0)))
        elif self.crop == "center":
            # Scale -> center crop
            transformer.append(torch_transforms.Resize(self.min_crop_size))
            transformer.append(torch_transforms.CenterCrop(self.crop_size))
        transformer.append(torch_transforms.ToTensor())
        transformer.append(torch_transforms.Normalize(
            mean=self._bgr_normalized_mean, std=self._bgr_normalized_std))
        self.transform = torch_transforms.Compose(transformer)

    def __getitem__(self, index):
        impath = self._imdb[index]["im_path"]
        target = self._imdb[index]["class"]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self._imdb)


"""DALI pipeline."""


class ImageList_DALI():

    def __init__(
            self,
            _list,
            batch_size,
            _rgb_normalized_mean=None,
            _rgb_normalized_std=None,
            crop='random',
            crop_size=224,
            min_crop_size=0.08,
            random_flip=False,
            num_workers=8,
            pin_memory=True,
            world_size=1,
            dali_cpu=True,
            temp_folder='/tmp'):
        logger.info("Using DALI as backend.")
        self._list = _list
        _rgb_mean = [i * 255. for i in _rgb_normalized_mean]
        _rgb_std = [i * 255. for i in _rgb_normalized_std]
        self.crop = crop
        self.crop_size = crop_size
        self.min_crop_size = min_crop_size
        self.random_flip = random_flip
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.world_size = world_size
        self.dali_cpu = dali_cpu
        # Save list to temp file
        _temp_save_file = os.path.join(
            temp_folder, random_time_string(8) + '.txt')
        logger.info("saving list to temp file")
        with open(_temp_save_file, "w") as f:
            for index, item in enumerate(self._list):
                f.write(item['im_path'] + ' ' + str(item['class']))
        # Construct pipeline
        device_id = torch.cuda.current_device()
        local_rank = torch.cuda.current_device()
        self.pipeline = ListPipe(batch_size, _temp_save_file, _rgb_mean, _rgb_std,
                                 data_dir='', crop=crop, crop_size=crop_size, min_crop_size=min_crop_size,
                                 random_flip=random_flip, device_id=device_id, num_threads=num_workers, local_rank=local_rank,
                                 world_size=world_size, dali_cpu=dali_cpu, shuffle=True)

    def get_data_iter(self):
        self.pipeline.build()
        return DaliIterator(pipelines=self.pipeline, size=self.get_size(
        ) / self.world_size)

    def get_size(self):
        return int(self.pipeline.epoch_size("Reader"))


class ListPipe(Pipeline):

    def __init__(self, batch_size, file_list, mean, std, data_dir='', crop='random', crop_size=224, min_crop_size=0.08,
                 random_flip=False, device_id=0, num_threads=8, local_rank=0, world_size=1, dali_cpu=False, shuffle=True):
        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(ListPipe, self).__init__(
            batch_size, num_threads, device_id, seed=-1)
        # Enabling read_ahead slowed down processing ~40%
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=shuffle, file_list=file_list)
        self.crop = crop
        self.random_flip = random_flip
        # Let user decide which pipeline works best with the chosen model
        if dali_cpu:
            decode_device = "cpu"
            self.dali_device = "cpu"
        else:
            decode_device = "mixed"
            self.dali_device = "gpu"
        # To be able to handle all images from full-sized ImageNet, this padding sets the size of the internal
        # nvJPEG buffers without additional reallocations
        device_memory_padding = 211025920 if decode_device == 'mixed' else 0
        host_memory_padding = 140544512 if decode_device == 'mixed' else 0
        if crop == 'random':
            self.decode = ops.ImageDecoderRandomCrop(device=decode_device, output_type=types.RGB,
                                                     device_memory_padding=device_memory_padding,
                                                     host_memory_padding=host_memory_padding,
                                                     random_aspect_ratio=[
                                                         0.8, 1.25],
                                                     random_area=[
                                                         min_crop_size, 1.0],
                                                     num_attempts=100)
            # Resize as desired.  To match torchvision data loader, use triangular interpolation.
            self.res = ops.Resize(device=self.dali_device, resize_x=crop_size, resize_y=crop_size,
                                  interp_type=types.INTERP_TRIANGULAR)

        else:
            self.decode = ops.ImageDecoder(device=decode_device, output_type=types.RGB,
                                           device_memory_padding=device_memory_padding,
                                           host_memory_padding=host_memory_padding)
            self.res = ops.Resize(
                device=self.dali_device, resize_shorter=min_crop_size, interp_type=types.INTERP_TRIANGULAR)
        self.cmn = ops.CropMirrorNormalize(device=self.dali_device,
                                           output_layout=types.NCHW,
                                           crop=(crop_size, crop_size),
                                           image_type=types.RGB,
                                           mean=mean,
                                           std=std,)
        if self.random_flip:
            self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        if self.random_flip:
            rng = self.coin()
            output = self.cmn(images, mirror=rng)
        else:
            output = self.cmn(images)
        self.labels = self.labels.gpu()
        return [output, self.labels]


class DaliIterator(object):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set

    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, pipelines, size, **kwargs):
        self._dali_iterator = DALIClassificationIterator(
            pipelines=pipelines, size=size)

    def gen_wrapper(dalipipeline):
        for data in dalipipeline:
            input = data[0]["data"]
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            yield input, target
        dalipipeline.reset()

    def __iter__(self):
        return DaliIterator.gen_wrapper(self._dali_iterator)

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))

    def reset(self):
        self._dali_iterator.reset()
