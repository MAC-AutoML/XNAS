import torchvision.datasets as dset
import datasets.preproc as preproc
import torchvision.transforms as transforms
import os
# try:
#     from nvidia.dali.plugin.pytorch import DALIClassificationIterator
#     from nvidia.dali.pipeline import Pipeline
#     import nvidia.dali.ops as ops
#     import nvidia.dali.types as types
# except ImportError:
#     raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
# except RuntimeError:
#     pass


def get_data(dataset, data_path, cutout_length, validation, image_size=None):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'cifar100':
        dset_cls = dset.CIFAR100
        n_classes = 100
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    elif 'imagenet' in dataset:
        dset_cls = dset.ImageFolder
        n_classes = 1000
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    if image_size is not None:
        trn_transform.transforms.append(transforms.Resize(image_size))
    if 'imagenet' in dataset:
        trn_data = dset_cls(root=os.path.join(data_path, 'train'), transform=trn_transform)
    else:
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    if 'imagenet' in dataset:
        if dataset == 'imagenet':
            shape = (len(trn_data.imgs), 224, 224, 3)
        elif dataset == 'imagenet112':
            shape = (len(trn_data.imgs), 112, 112, 3)
        elif dataset == 'imagenet56':
            shape = (len(trn_data.imgs), 56, 56, 3)
        else:
            raise NotImplementedError
    else:
        if hasattr(trn_data, 'data'):
            shape = trn_data.data.shape
        else:
            shape = trn_data.train_data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1] if image_size is None else image_size

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        if dataset == 'imagenet':
            ret.append(dset_cls(root=os.path.join(data_path,'val'), transform=val_transform))
        else:
            ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))
    return ret


# class HybridTrainPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
#         super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         self.input = ops.FileReader(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=True)
#         #let user decide which pipeline works him bets for RN version he runs
#         if dali_cpu:
#             dali_device = "cpu"
#             self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
#                                                     random_aspect_ratio=[0.8, 1.25],
#                                                     random_area=[0.1, 1.0],
#                                                     num_attempts=100)
#         else:
#             dali_device = "gpu"
#             # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
#             # without additional reallocations
#             self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB,
#                                                       device_memory_padding=211025920,
#                                                       host_memory_padding=140544512,
#                                                       random_aspect_ratio=[0.8, 1.25],
#                                                       random_area=[0.1, 1.0],
#                                                       num_attempts=100)
#         self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             crop=(crop, crop),
#                                             image_type=types.RGB,
#                                             mean=[0.485 * 255,0.456 * 255,0.406 * 255],
#                                             std=[0.229 * 255,0.224 * 255,0.225 * 255])
#         self.coin = ops.CoinFlip(probability=0.5)
#         # self.color_jitter = [ops.Brightness(device="gpu", brightness=0.4),
#         #                      ops.Contrast(device="gpu", contrast=0.4),
#         #                      ops.Saturation(device="gpu", saturation=0.4),
#         #                      ops.Hue(device="gpu", hue=0.2)]
#         # print('DALI "{0}" variant'.format(dali_device))
#
#     def define_graph(self):
#         rng = self.coin()
#         self.jpegs, self.labels = self.input(name="Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         # for color_op in self.color_jitter:
#         #     images = color_op(images.gpu())
#         output = self.cmnp(images.gpu(), mirror=rng)
#         return [output, self.labels]
#
#
# class HybridValPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
#         super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         self.input = ops.FileReader(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=False)
#         self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
#         self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             crop=(crop, crop),
#                                             image_type=types.RGB,
#                                             mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
#
#     def define_graph(self):
#         self.jpegs, self.labels = self.input(name="Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         output = self.cmnp(images)
#         return [output, self.labels]
#
#
# def get_dali_imagenet_pipeline(batch_size, num_threads, data_path, train_cpu=False,
#                                crop=224, size=256):
#     train_pipe = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=0,
#                            data_dir=os.path.join(data_path, 'train'),
#                            crop=crop, dali_cpu=train_cpu)
#     train_pipe.build()
#
#     val_pipe = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=0,
#                          data_dir=os.path.join(data_path, 'val'),
#                          crop=crop, size=size)
#     val_pipe.build()
#     return [train_pipe, val_pipe]

