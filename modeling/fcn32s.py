# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from layers.bilinear_upsample import bilinear_upsampling
from layers.conv_layer import conv_layer
from .backbone import build_backbone


class FCN32s(nn.Module):
    def __init__(self, cfg):
        super(FCN32s, self).__init__()
        self.backbone = build_backbone(cfg)
        num_classes = cfg.MODEL.NUM_CLASSES

        self.fc1 = conv_layer(512, 4096, 7)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d()

        self.fc2 = conv_layer(4096, 4096, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d()

        self.score_fr = conv_layer(4096, num_classes, 1)
        self.upscore = bilinear_upsampling(num_classes, num_classes, 64, stride=32,
                                           bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.backbone(x)
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)

        x = self.relu2(self.fc2(x))
        x = self.drop2(x)

        x = self.score_fr(x)
        x = self.upscore(x)
        x = x[:, :, 19:19 + h, 19:19 + w].contiguous()
        return x

    def copy_params_from_vgg16(self, vgg16):
        feat = self.backbone
        features = [
            feat.conv1_1, feat.relu1_1,
            feat.conv1_2, feat.relu1_2,
            feat.pool1,
            feat.conv2_1, feat.relu2_1,
            feat.conv2_2, feat.relu2_2,
            feat.pool2,
            feat.conv3_1, feat.relu3_1,
            feat.conv3_2, feat.relu3_2,
            feat.conv3_3, feat.relu3_3,
            feat.pool3,
            feat.conv4_1, feat.relu4_1,
            feat.conv4_2, feat.relu4_2,
            feat.conv4_3, feat.relu4_3,
            feat.pool4,
            feat.conv5_1, feat.relu5_1,
            feat.conv5_2, feat.relu5_2,
            feat.conv5_3, feat.relu5_3,
            feat.pool5
        ]

        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc1', 'fc2']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
