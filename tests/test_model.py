import sys
import unittest

sys.path.append('.')
from modeling.backbones.vgg import VGG16
from config import cfg
from modeling import build_fcn_model
import torch


class MyTestCase(unittest.TestCase):
    def test_vgg(self):
        model = build_fcn_model(cfg)
        # x = torch.randn(5, 3, 224, 224)
        # y = model(x)
        from IPython import embed;
        embed()


if __name__ == '__main__':
    unittest.main()
