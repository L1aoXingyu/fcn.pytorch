# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from torch import nn


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Make a 2D bilinear kernel suitable for unsampling
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight).float()


def bilinear_upsampling(in_channels, out_channels, kernel_size, stride, bias=False):
    initial_weight = get_upsampling_weight(in_channels, out_channels, kernel_size)
    layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
    layer.weight.data.copy_(initial_weight)
    # weight is frozen because it's just a bilinear upsampling
    layer.weight.requires_grad = False
    return layer
