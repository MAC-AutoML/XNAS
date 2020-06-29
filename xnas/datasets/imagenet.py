#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import gc
import importlib
import os
import re
import time

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as torch_transforms
from torch.utils.data.distributed import DistributedSampler

import xnas.core.logging as logging
import xnas.datasets.transforms as custom_transforms
from xnas.core.config import cfg

try:
    from nvidia import dali
    from pytorch_cls.datasets.dali import HybridTrainPipe, HybridValPipe, DaliIterator
except ImportError:
    print('Could not import DALI')

logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order, only used in ImageNet custom backend
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)


def ImageNet(data_path, split,  batch_size, shuffle, drop_last):
    assert cfg.DATA_LOADER.BACKEND in [
        'custom', 'dali_cpu', 'dali_gpu', 'torch']
    if cfg.DATA_LOADER.BACKEND == 'custom':
        dataset = ImageNet_custom(data_path, split)
        # Create a sampler for multi-process training
        sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
        )
        return loader
    else:
        use_dali = True if 'dali' in cfg.DATA_LOADER.BACKEND else False
        use_dali_cpu = True if cfg.DATA_LOADER.BACKEND == 'dali_cpu' else False
        # In torch cuda amp, we do not manually change data to FP16
        # use_fp16 = (split == 'train' & cfg.TRAIN.FP16) | (split == 'val' & cfg.TEST.FP16)
        dataset = ImageNet_(data_path,
                            batch_size=batch_size,
                            size=cfg.TRAIN.IM_SIZE,
                            val_batch_size=cfg.TEST.BATCH_SIZE,
                            val_size=cfg.TEST.IM_SIZE,
                            min_crop_size=0.08,
                            workers=cfg.DATA_LOADER.NUM_WORKERS,
                            world_size=cfg.DATA_LOADER.WORLD_SIZE,
                            cuda=True,
                            use_dali=use_dali,
                            dali_cpu=use_dali_cpu,
                            fp16=False,
                            mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                            std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
                            pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
        if split == 'train':
            return dataset.train_loader
        elif split == 'val':
            return dataset.val_loader
        else:
            raise NotImplementedError


class ImageNet_custom(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "val"]
        assert split in splits, "Split '{}' not supported for ImageNet".format(
            split)
        logger.info("Constructing ImageNet {}...".format(split))
        self._data_path, self._split = data_path, split
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self._data_path, self._split)
        logger.info("{} data path: {}".format(self._split, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        split_files = os.listdir(split_path)
        self._class_ids = sorted(
            f for f in split_files if re.match(r"^n[0-9]+$", f))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        # Train and test setups differ
        train_size = cfg.TRAIN.IM_SIZE
        if self._split == "train":
            # Scale and aspect ratio then horizontal flip
            im = custom_transforms.random_sized_crop(
                im=im, size=train_size, area_frac=0.08)
            im = custom_transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        else:
            # Scale and center crop
            im = custom_transforms.scale(cfg.TEST.IM_SIZE, im)
            im = custom_transforms.center_crop(train_size, im)
        # HWC -> CHW
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # PCA jitter
        if self._split == "train":
            if cfg.DATA_LOADER.PCA_JITTER:
                im = custom_transforms.lighting(im, 0.1, _EIG_VALS, _EIG_VECS)
            if cfg.DATA_LOADER.COLOR_JITTER:
                raise NotImplementedError
        # Color normalization
        im = custom_transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        # Load the image
        im = cv2.imread(self._imdb[index]["im_path"])
        im = im.astype(np.float32, copy=False)
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)


def clear_memory(verbose=False):
    stt = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
    gc.collect()

    if verbose:
        print('Cleared memory.  Time taken was %f secs' % (time.time() - stt))


"""
ImageNet on DALI
"""


class ImageNet_():
    """
    Pytorch Dataloader, with torchvision or Nvidia DALI CPU/GPU pipelines.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip
    And ImageNet style validation preprocessing, namely:
    -resize to specified size
    -center crop to desired size
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    batch_size (int): how many samples per batch to load
    size (int): Output size (typically 224 for ImageNet)
    val_size (int): Validation pipeline resize size (typically 256 for ImageNet)
    workers (int): how many workers to use for data loading
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    cuda (bool): Output tensors on CUDA, CPU otherwise
    use_dali (bool): Use Nvidia DALI backend, torchvision otherwise
    dali_cpu (bool): Use Nvidia DALI cpu backend, GPU backend otherwise
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer CPU tensor to pinned memory before transfer to GPU (torchvision only)
    pin_memory_dali (bool): Transfer CPU tensor to pinned memory before transfer to GPU (dali only)
    """

    def __init__(self,
                 data_dir,
                 batch_size=256,
                 size=224,
                 val_batch_size=200,
                 val_size=256,
                 min_crop_size=0.08,
                 workers=8,
                 world_size=1,
                 cuda=True,
                 use_dali=False,
                 dali_cpu=False,
                 fp16=False,
                 mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                 std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
                 pin_memory=True,
                 color_jitter=False
                 ):

        self.batch_size = batch_size
        self.size = size
        self.val_batch_size = val_batch_size
        self.min_crop_size = min_crop_size
        self.workers = workers
        self.world_size = world_size
        self.cuda = cuda
        self.use_dali = use_dali
        self.dali_cpu = dali_cpu
        self.fp16 = fp16
        self.mean = mean
        self.std = std
        self.pin_memory = pin_memory
        self.color_jitter = color_jitter

        # color jitter not implenment yet since it may have a worse performnace in imagenet
        # according https://github.com/pytorch/examples/issues/291 and https://github.com/pytorch/examples/blob/master/imagenet/main.py
        if self.color_jitter:
            raise NotImplementedError

        self.val_size = val_size
        if self.val_size is None:
            self.val_size = self.size

        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size

        # Data loading code
        self.traindir = os.path.join(data_dir, 'train')
        self.valdir = os.path.join(data_dir, 'val')
        # DALI Dataloader
        if self.use_dali:
            logger.info('Using Nvidia DALI dataloader')
            assert len(datasets.ImageFolder(
                self.valdir)) % self.val_batch_size == 0, 'Validation batch size must divide validation dataset size cleanly...  DALI has problems otherwise.'
            self._build_dali_pipeline()

        # Standard torchvision dataloader
        else:
            logger.info('Using torchvision dataloader')
            self._build_torchvision_pipeline()

    def _build_torchvision_pipeline(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        preproc_train = [torch_transforms.RandomResizedCrop(self.size, scale=(self.min_crop_size, 1.0)),
                         torch_transforms.RandomHorizontalFlip(),
                         torch_transforms.ToTensor(),
                         torch_transforms.Normalize(
                             mean=self.mean, std=self.std),
                         ]

        preproc_val = [torch_transforms.Resize(self.val_size),
                       torch_transforms.CenterCrop(self.size),
                       torch_transforms.ToTensor(),
                       torch_transforms.Normalize(
                           mean=self.mean, std=self.std),
                       ]

        train_dataset = datasets.ImageFolder(
            self.traindir, torch_transforms.Compose(preproc_train))
        val_dataset = datasets.ImageFolder(
            self.valdir, torch_transforms.Compose(preproc_val))

        self.train_sampler = None
        self.val_sampler = None

        if cfg.NUM_GPUS > 1:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=(
                self.train_sampler is None),
            num_workers=self.workers, pin_memory=self.pin_memory, sampler=self.train_sampler)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.pin_memory, sampler=self.val_sampler)

    def _build_dali_pipeline(self, val_on_cpu=True):
        current_device = torch.cuda.current_device()
        # assert self.world_size == 1, 'Distributed support not tested yet'

        iterator_train = DaliIterator

        self.train_pipe = HybridTrainPipe(batch_size=self.batch_size, num_threads=self.workers, device_id=current_device,
                                          data_dir=self.traindir, crop=self.size, dali_cpu=self.dali_cpu,
                                          mean=self.mean, std=self.std, local_rank=current_device,
                                          world_size=self.world_size, shuffle=True, fp16=self.fp16, min_crop_size=self.min_crop_size)

        self.train_pipe.build()
        self.train_loader = iterator_train(pipelines=self.train_pipe, size=self.get_nb_train(
        ) / self.world_size)

        iterator_val = DaliIterator

        self.val_pipe = HybridValPipe(batch_size=self.val_batch_size, num_threads=self.workers, device_id=current_device,
                                      data_dir=self.valdir, crop=self.size, size=self.val_size, dali_cpu=val_on_cpu,
                                      mean=self.mean, std=self.std, local_rank=current_device,
                                      world_size=self.world_size, shuffle=False, fp16=self.fp16)

        self.val_pipe.build()
        self.val_loader = iterator_val(pipelines=self.val_pipe, size=self.get_nb_val(
        ) / self.world_size)

    def _get_torchvision_loader(self, loader):
        return TorchvisionIterator(loader=loader,
                                   cuda=self.cuda,
                                   fp16=self.fp16,
                                   mean=self.mean,
                                   std=self.std,
                                   )

    def get_train_loader(self):
        """
        Creates & returns an iterator for the training dataset
        :return: Dataset iterator object
        """
        if self.use_dali:
            return self.train_loader
        return self._get_torchvision_loader(loader=self.train_loader)

    def get_val_loader(self):
        """
        Creates & returns an iterator for the training dataset
        :return: Dataset iterator object
        """
        if self.use_dali:
            return self.val_loader
        return self._get_torchvision_loader(loader=self.val_loader)

    def get_nb_train(self):
        """
        :return: Number of training examples
        """
        if self.use_dali:
            return int(self.train_pipe.epoch_size("Reader"))
        return len(datasets.ImageFolder(self.traindir))

    def get_nb_val(self):
        """
        :return: Number of validation examples
        """
        if self.use_dali:
            return int(self.val_pipe.epoch_size("Reader"))
        return len(datasets.ImageFolder(self.valdir))

    def prep_for_val(self):
        self.reset(val_on_cpu=False)

    # This is needed only for DALI
    def reset(self, val_on_cpu=True):
        if self.use_dali:
            self.train_loader._dali_iterator.reset()
            self.val_loader._dali_iterator.reset()
            clear_memory()

            # # Currently we need to delete & rebuild the dali pipeline every epoch,
            # # due to a memory leak somewhere in DALI
            # logger.info('Recreating DALI dataloaders to reduce memory usage')
            # del self.train_loader, self.val_loader, self.train_pipe, self.val_pipe
            # clear_memory()

            # # taken from: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive
            # importlib.reload(dali)
            # from pycls.datasets.dali import HybridTrainPipe, HybridValPipe, DaliIteratorCPU, DaliIteratorGPU

            # self._build_dali_pipeline(val_on_cpu=val_on_cpu)

    def set_train_batch_size(self, train_batch_size):
        self.batch_size = train_batch_size
        if self.use_dali:
            del self.train_loader, self.val_loader, self.train_pipe, self.val_pipe
            self._build_dali_pipeline()
        else:
            del self.train_sampler, self.val_sampler, self.train_loader, self.val_loader
            self._build_torchvision_pipeline()

    def get_nb_classes(self):
        """
        :return: The number of classes in the dataset - as indicated by the validation dataset
        """
        return len(datasets.ImageFolder(self.valdir).classes)


def fast_collate(batch):
    """Convert batch into tuple of X and Y tensors."""
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


class TorchvisionIterator():
    """
    Iterator to perform final data pre-processing steps:
    -transfer to device (done on 8 bit tensor to reduce bandwidth requirements)
    -convert to fp32/fp16 tensor
    -apply mean/std scaling
    loader (DataLoader): Torchvision Dataloader
    cuda (bool): Transfer tensor to CUDA device
    fp16 (bool): Convert tensor to fp16 instead of fp32
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    """

    def __init__(self,
                 loader,
                 cuda=False,
                 fp16=False,
                 mean=(0., 0., 0.),
                 std=(1., 1., 1.),
                 ):
        logger.info('Using Torchvision iterator')
        self.loader = iter(loader)
        self.cuda = cuda
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.fp16 = fp16

        if self.cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __iter__(self):
        return self

    def __next__(self):
        input, target = next(self.loader)

        if self.cuda:
            input = input.cuda()
            target = target.cuda()

        if self.fp16:
            input = input.half()
        else:
            input = input.float()

        input = input.sub_(self.mean).div_(self.std)

        return input, target

    def __len__(self):
        return len(self.loader)
