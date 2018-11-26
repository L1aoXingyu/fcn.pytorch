# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
import torchvision.transforms as T

from .transforms import RandomCrop, RandomHorizontalFlip, image2label


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        def transform(img, target):
            if cfg.INPUT.IS_CROP:
                img, target = RandomCrop(cfg.INPUT.SIZE_TRAIN)(img, target)
            img, target = RandomHorizontalFlip(cfg.INPUT.PROB)(img, target)
            # convert image to tensor
            img = T.ToTensor()(img)
            img = normalize_transform(img)
            # convert target to label
            label = image2label(target)
            label = torch.from_numpy(label)
            return img, label

        return transform
    else:
        def transform(img, target):
            img = T.ToTensor()(img)
            img = normalize_transform(img)
            label = image2label(target)
            label = torch.from_numpy(label)
            return img, label
        return transform
