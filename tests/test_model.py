import sys
import unittest

sys.path.append('.')
from modeling.backbones.vgg import VGG16
from modeling.fcn32s import FCN32s
import torch


class MyTestCase(unittest.TestCase):
    def test_vgg(self):
        model = FCN32s(VGG16(pretrained=True), 21)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)


if __name__ == '__main__':
    unittest.main()
