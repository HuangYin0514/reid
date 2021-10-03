# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(opt, is_train=True):

    T_image_size = (opt.img_height, opt.img_width)
    T_mean = [0.485, 0.456, 0.406]
    T_std = [0.229, 0.224, 0.225]

    normalize_transform = T.Normalize(mean=T_mean, std=T_std)
    if is_train:
        transform = T.Compose(
            [
                T.Resize(T_image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(10),
                T.RandomCrop(T_image_size),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=0.5, mean=T_mean),
            ]
        )
    else:
        transform = T.Compose(
            [T.Resize(T_image_size), T.ToTensor(), normalize_transform]
        )

    return transform
