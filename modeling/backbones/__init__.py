# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from .vgg import VGG16


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.NAME == 'vgg16':
        feature, classifier = VGG16(cfg)
        return feature, classifier
