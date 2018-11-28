# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from .vgg import VGG16


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.NAME == 'vgg16':
        backbone = VGG16()
        return backbone
