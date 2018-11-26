# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys
import unittest

sys.path.append('.')
from config import cfg
from data.transforms import build_transforms
from data.build import build_dataset
from solver.build import make_optimizer


class TestDataSet(unittest.TestCase):
    def test_dataset(self):
        train_transform = build_transforms(cfg, True)
        val_transform = build_transforms(cfg, False)
        train_set = build_dataset(cfg, train_transform, True)
        val_test = build_dataset(cfg, val_transform, False)
        from IPython import embed;
        embed()


if __name__ == '__main__':
    unittest.main()
