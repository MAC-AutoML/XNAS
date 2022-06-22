import math
import torch
from PIL import Image
import torchvision.transforms as transforms

from xnas.core.config import cfg
from xnas.datasets.auto_augment_tf import auto_augment_policy, AutoAugment


IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD = [0.229, 0.224, 0.225]


def get_data_transform(augment, **kwargs):
    if len(cfg.SEARCH.MULTI_SIZES)==0:
        # using single image_size for training
        train_crop_size = cfg.SEARCH.IM_SIZE
    else:
        # using MultiSize_RandomCrop
        train_crop_size = cfg.SEARCH.MULTI_SIZES
    min_train_scale = 0.08
    test_scale = math.ceil(cfg.TEST.IM_SIZE / 0.875)    # 224 / 0.875 = 256
    test_crop_size = cfg.TEST.IM_SIZE   # do not crop and using 224 by default.

    interpolation = transforms.InterpolationMode.BICUBIC
    if 'interpolation' in kwargs.keys() and kwargs['interpolation'] == 'bilinear':
        interpolation = transforms.InterpolationMode.BILINEAR
    
    da_args = {
        'train_crop_size': train_crop_size,
        'train_min_scale': min_train_scale,
        'test_scale': test_scale,
        'test_crop_size': test_crop_size,
        'interpolation': interpolation,
    }

    if augment == 'default':
        return build_default_transform(**da_args)
    elif augment == 'auto_augment_tf':
        policy = 'v0' if 'policy' not in kwargs.keys() else kwargs['policy']
        return build_imagenet_auto_augment_tf_transform(policy=policy, **da_args)
    else:
        raise ValueError(augment)


def get_normalize():
    return transforms.Normalize(
        mean=torch.Tensor(IMAGENET_RGB_MEAN),
        std=torch.Tensor(IMAGENET_RGB_STD),
    )


def get_randomResizedCrop(train_crop_size=224, train_min_scale=0.08, interpolation=transforms.InterpolationMode.BICUBIC):
    if isinstance(train_crop_size, int):
        return transforms.RandomResizedCrop(train_crop_size, scale=(train_min_scale, 1.0), interpolation=interpolation)
    elif isinstance(train_crop_size, list):
        from xnas.datasets.transforms import MultiSizeRandomCrop
        msrc = MultiSizeRandomCrop(train_crop_size)
        return msrc
    else:
        raise TypeError(train_crop_size)


def build_default_transform(
    train_crop_size=224, train_min_scale=0.08, test_scale=256, test_crop_size=224, interpolation=transforms.InterpolationMode.BICUBIC
):
    normalize = get_normalize()
    train_crop_transform = get_randomResizedCrop(
        train_crop_size, train_min_scale, interpolation
    )
    train_transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
            train_crop_transform,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(test_scale, interpolation=interpolation),
            transforms.CenterCrop(test_crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, test_transform


def build_imagenet_auto_augment_tf_transform(
    policy='v0', train_crop_size=224, train_min_scale=0.08, test_scale=256, test_crop_size=224, interpolation=transforms.InterpolationMode.BICUBIC
):

    normalize = get_normalize()
    img_size = train_crop_size
    aa_params = {
        "translate_const": int(img_size * 0.45),
        "img_mean": tuple(round(x) for x in IMAGENET_RGB_MEAN),
    }

    aa_policy = AutoAugment(auto_augment_policy(policy, aa_params))
    train_crop_transform = get_randomResizedCrop(
        train_crop_size, train_min_scale, interpolation
    )
    train_transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
            train_crop_transform,
            transforms.RandomHorizontalFlip(),
            aa_policy,
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(test_scale, interpolation=interpolation),
            transforms.CenterCrop(test_crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, test_transform
