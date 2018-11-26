# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .fcn32s import FCN32s

_FCN_META_ARCHITECTURE = {'fcn32s': FCN32s}


def build_fcn_model(cfg):
    meta_arch = _FCN_META_ARCHITECTURE[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
