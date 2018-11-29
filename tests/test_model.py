import sys
import unittest

sys.path.append('.')
from modeling.backbone.vgg import VGG16
from config import cfg
from modeling import build_fcn_model
from modeling.backbone import build_backbone
import torch


class MyTestCase(unittest.TestCase):
    def test_vgg(self):
        vgg = build_backbone(cfg)
        model = build_fcn_model(cfg)
        print(model.backbone.conv1_1.weight[0, 0, 0, 0])
        # x = torch.randn(5, 3, 224, 224)
        # y = model(x)
        from IPython import embed;
        embed()


if __name__ == '__main__':
    unittest.main()
