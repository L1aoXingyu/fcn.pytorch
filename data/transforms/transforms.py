# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import random

import numpy as np
import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(target)
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def image2label(img):
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(COLORMAP):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    data = np.array(img, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1] * 256 + data[:, :, 2])
    return np.array(cm2lbl[idx], dtype=np.int64)


CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class.
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
