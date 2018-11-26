# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os

from PIL import Image
from torch.utils import data


def read_images(root, train):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/') + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    return data, label


class VocSegDataset(data.Dataset):
    def __init__(self, cfg, train, transforms=None):
        self.cfg = cfg
        self.train = train
        self.transforms = transforms
        self.data_list, self.label_list = read_images(self.cfg.DATASETS.ROOT, train)
        if cfg.INPUT.IS_CROP:
            self.data_list = self._filter(self.data_list)
            self.label_list = self._filter(self.label_list)

    def _filter(self, images):
        return [img for img in images if (Image.open(img).size[1] >= self.cfg.INPUT.SIZE_TRAIN[0] and
                                          Image.open(img).size[0] >= self.cfg.INPUT.SIZE_TRAIN[1])]

    def __getitem__(self, item):
        img = self.data_list[item]
        label = self.label_list[item]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label)
        return img, label

    def __len__(self):
        return len(self.data_list)
