# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch

from .backbone.vgg import pretrained_vgg
from .fcn16s import FCN16s
from .fcn32s import FCN32s
from .fcn8s import FCN8s

_FCN_META_ARCHITECTURE = {'fcn32s': FCN32s,
                          'fcn16s': FCN16s,
                          'fcn8s': FCN8s}


def build_fcn_model(cfg):
    meta_arch = _FCN_META_ARCHITECTURE[cfg.MODEL.META_ARCHITECTURE]
    model = meta_arch(cfg)
    if cfg.MODEL.BACKBONE.PRETRAINED:
        vgg16 = pretrained_vgg(cfg)
        model.copy_params_from_vgg16(vgg16)
    if cfg.MODEL.REFINEMENT.NAME == 'fcn32s':
        fcn32s = FCN32s(cfg)
        fcn32s.load_state_dict(torch.load(cfg.MODEL.REFINEMENT.WEIGHT))
        model.copy_params_from_fcn32s(fcn32s)
    elif cfg.MODEL.REFINEMENT.NAME == 'fcn16s':
        fcn16s = FCN16s(cfg)
        fcn16s.load_state_dict(torch.load(cfg.MODEL.REFINEMENT.WEIGHT))
        model.copy_params_from_fcn16s(fcn16s)
    return model
