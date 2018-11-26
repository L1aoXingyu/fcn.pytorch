# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from layers.bilinear_upsample import bilinear_upsampling
from .backbones import build_backbone


class FCN32s(nn.Module):
    def __init__(self, cfg):
        super(FCN32s, self).__init__()
        self.feature, classifier = build_backbone(cfg)
        num_classes = cfg.MODEL.NUM_CLASSES

        self.fc1 = nn.Conv2d(512, 4096, 7)
        self.fc1.weight.data.copy_(classifier[0].weight.view(self.fc1.weight.size()))
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d()

        self.fc2 = nn.Conv2d(4096, 4096, 1)
        self.fc2.weight.data.copy_(classifier[3].weight.view(self.fc2.weight.size()))
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.upscore = bilinear_upsampling(num_classes, num_classes, 64, stride=32,
                                           bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.feature(x)
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)

        x = self.relu2(self.fc2(x))
        x = self.drop2(x)

        x = self.score_fr(x)
        x = self.upscore(x)
        x = x[:, :, 19:19 + h, 19:19 + w].contiguous()
        return x
