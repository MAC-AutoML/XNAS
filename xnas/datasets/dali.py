"""
Copyed from https://github.com/yaysummeriscoming/DALI_pytorch_demo/edit/master/dali.py
"""
import torch
import math

import threading
from torch.multiprocessing import Event
from torch._six import queue

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    """
    DALI Train Pipeline
    Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
    In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip

    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    local_rank (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
    min_crop_size (float, optional, default = 0.08) - Minimum random crop size
    """

    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 mean, std, local_rank=0, world_size=1, dali_cpu=False, shuffle=True, fp16=False,
                 min_crop_size=0.08, color_jitter=False):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=-1)

        # Enabling read_ahead slowed down processing ~40%
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=shuffle)

        # Let user decide which pipeline works best with the chosen model
        if dali_cpu:
            decode_device = "cpu"
            self.dali_device = "cpu"
            self.flip = ops.Flip(device=self.dali_device)
        else:
            decode_device = "mixed"
            self.dali_device = "gpu"

        output_dtype = types.FLOAT
        if fp16:
            output_dtype = types.FLOAT16

        self.cmn = ops.CropMirrorNormalize(device=self.dali_device,
                                           output_dtype=output_dtype,
                                           output_layout=types.NCHW,
                                           crop=(crop, crop),
                                           image_type=types.RGB,
                                           mean=mean,
                                           std=std,)

        # To be able to handle all images from full-sized ImageNet, this padding sets the size of the internal
        # nvJPEG buffers without additional reallocations
        device_memory_padding = 211025920 if decode_device == 'mixed' else 0
        host_memory_padding = 140544512 if decode_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decode_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[
                                                     0.8, 1.25],
                                                 random_area=[
                                                     min_crop_size, 1.0],
                                                 num_attempts=100)

        # Resize as desired.  To match torchvision data loader, use triangular interpolation.
        self.res = ops.Resize(device=self.dali_device, resize_x=crop, resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)

        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(self.dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        # Combined decode & random crop
        images = self.decode(self.jpegs)

        # Resize as desired
        images = self.res(images)
        output = self.cmn(images, mirror=rng)
        self.labels = self.labels.gpu()
        return [output, self.labels]


class HybridValPipe(Pipeline):
    """
    DALI Validation Pipeline
    Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
    In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
    This dataloader implements ImageNet style validation preprocessing, namely:
    -resize to specified size
    -center crop to desired size

    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
        containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    size (int): Resize size (typically 256 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    local_rank (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
    """

    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size,
                 mean, std, local_rank=0, world_size=1, dali_cpu=False, shuffle=False, fp16=False):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(HybridValPipe, self).__init__(
            batch_size, num_threads, device_id, seed=-1)

        # Enabling read_ahead slowed down processing ~40%
        # Note: initial_fill is for the shuffle buffer.  As we only want to see every example once, this is set to 1
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank,
                                    num_shards=world_size, random_shuffle=shuffle, initial_fill=1)
        if dali_cpu:
            decode_device = "cpu"
            self.dali_device = "cpu"
            self.crop = ops.Crop(device="cpu", crop=(crop, crop))

        else:
            decode_device = "mixed"
            self.dali_device = "gpu"

        output_dtype = types.FLOAT
        if fp16:
            output_dtype = types.FLOAT16

        self.cmnp = ops.CropMirrorNormalize(device=self.dali_device,
                                            output_dtype=output_dtype,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)

        self.decode = ops.ImageDecoder(
            device=decode_device, output_type=types.RGB)

        # Resize to desired size.  To match torchvision dataloader, use triangular interpolation
        self.res = ops.Resize(
            device=self.dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        self.labels = self.labels.gpu()
        return [output, self.labels]


def expand(num_classes, dtype, tensor):
    e = torch.zeros(tensor.size(0), num_classes,
                    dtype=dtype, device=torch.device('cuda'))
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


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
