# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_fcn_model
from solver import make_optimizer
from utils.logger import setup_logger
from layers.cross_entropy2d import cross_entropy2d


def train(cfg):
    model = build_fcn_model(cfg)

    optimizer = make_optimizer(cfg, model)

    arguments = {}

    data_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    do_train(
        cfg,
        model,
        data_loader,
        val_loader,
        optimizer,
        cross_entropy2d,
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch FCN Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("FCN_Model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg)


if __name__ == '__main__':
    main()
