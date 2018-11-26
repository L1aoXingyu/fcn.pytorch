# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch
import torchvision


def VGG16(cfg):
    model = torchvision.models.vgg16()
    if cfg.MODEL.BACKBONE.PRETRAINED:
        model.load_state_dict(torch.load(cfg.MODEL.BACKBONE.LOAD_PATH))
    features = model.features
    classifier = model.classifier
    features[0].padding = (100, 100)
    for m in features:
        if m.__class__.__name__.find('MaxPool') != -1:
            m.ceil_mode = True
    return features, classifier
