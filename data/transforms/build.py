# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
import torchvision.transforms as T

from .transforms import RandomHorizontalFlip


def build_transforms(cfg, is_train=True):
    normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        def transform(img, target):
            img, target = RandomHorizontalFlip(cfg.INPUT.PROB)(img, target)
            img = T.ToTensor()(img)
            img = normalize(img)
            # label = image2label(target)
            label = np.array(target, dtype=np.int64)
            # remove boundary
            label[label == 255] = -1
            label = torch.from_numpy(label)
            return img, label

        return transform
    else:
        def transform(img, target):
            img = T.ToTensor()(img)
            img = normalize(img)
            # label = image2label(target)
            label = np.array(target, dtype=np.int64)
            # remove boundary
            label[label == 255] = -1
            label = torch.from_numpy(label)
            return img, label

        return transform


def build_untransform(cfg):
    def untransform(img, target):
        img = img * torch.FloatTensor(cfg.INPUT.PIXEL_STD)[:, None, None] \
              + torch.FloatTensor(cfg.INPUT.PIXEL_MEAN)[:, None, None]
        origin_img = torch.clamp(img, min=0, max=1) * 255
        origin_img = origin_img.permute(1, 2, 0).numpy()
        origin_img = origin_img.astype(np.uint8)

        label = target.numpy()
        label[label == -1] = 0
        return origin_img, label

    return untransform
