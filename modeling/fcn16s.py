# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from layers.bilinear_upsample import bilinear_upsampling
from layers.conv_layer import conv_layer
from .backbone import build_backbone


class FCN16s(nn.Module):
    def __init__(self, cfg):
        super(FCN16s, self).__init__()
        self.backbone = build_backbone(cfg)
        num_classes = cfg.MODEL.NUM_CLASSES

        # fc1
        self.fc1 = conv_layer(512, 4096, 7)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d()

        # fc2
        self.fc2 = conv_layer(4096, 4096, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d()

        self.score_fr = conv_layer(4096, num_classes, 1)
        self.score_pool4 = conv_layer(512, num_classes, 1)

        self.upscore2 = bilinear_upsampling(num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore16 = bilinear_upsampling(num_classes, num_classes, 32, stride=16, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.backbone.conv1_1(x)
        x = self.backbone.relu1_1(x)
        x = self.backbone.conv1_2(x)
        x = self.backbone.relu1_2(x)
        x = self.backbone.pool1(x)

        x = self.backbone.conv2_1(x)
        x = self.backbone.relu2_1(x)
        x = self.backbone.conv2_2(x)
        x = self.backbone.relu2_2(x)
        x = self.backbone.pool2(x)

        x = self.backbone.conv3_1(x)
        x = self.backbone.relu3_1(x)
        x = self.backbone.conv3_2(x)
        x = self.backbone.relu3_2(x)
        x = self.backbone.conv3_3(x)
        x = self.backbone.relu3_3(x)
        x = self.backbone.pool3(x)

        x = self.backbone.conv4_1(x)
        x = self.backbone.relu4_1(x)
        x = self.backbone.conv4_2(x)
        x = self.backbone.relu4_2(x)
        x = self.backbone.conv4_3(x)
        x = self.backbone.relu4_3(x)
        x = self.backbone.pool4(x)
        pool4 = x  # 1/16

        x = self.backbone.conv5_1(x)
        x = self.backbone.relu5_1(x)
        x = self.backbone.conv5_2(x)
        x = self.backbone.relu5_2(x)
        x = self.backbone.conv5_3(x)
        x = self.backbone.relu5_3(x)
        x = self.backbone.pool5(x)

        x = self.relu1(self.fc1(x))
        x = self.drop1(x)

        x = self.relu2(self.fc2(x))
        x = self.drop2(x)

        x = self.score_fr(x)
        x = self.upscore2(x)
        upscore2 = x

        x = self.score_pool4(pool4)
        x = x[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = x  # 1/16

        x = upscore2 + score_pool4c

        x = self.upscore16(x)
        x = x[:, :, 27:27 + h, 27:27 + w].contiguous()
        return x

    def copy_params_from_fcn32s(self, fcn32s):
        # load backbone
        self.backbone.load_state_dict(fcn32s.backbone.state_dict())
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except AttributeError:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)


