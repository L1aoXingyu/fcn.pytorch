# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.engine import Engine, Events
from tensorboardX import SummaryWriter

from data.transforms import build_untransform
from data.transforms.transforms import COLORMAP
from utils.metric import Label_Accuracy

plt.switch_backend('agg')


def create_evaluator(model, metrics={}, device=None):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x = x.to(device)
            y_pred = model(x)
            return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader
):
    cm = np.array(COLORMAP).astype(np.uint8)
    untransform = build_untransform(cfg)

    device = cfg.MODEL.DEVICE
    output_dir = cfg.OUTPUT_DIR

    logger = logging.getLogger("FCN_Model.inference")
    logger.info("Start inferencing")
    evaluator = create_evaluator(model, metrics={'mean_iu': Label_Accuracy(cfg.MODEL.NUM_CLASSES)}, device=device)

    writer = SummaryWriter(output_dir + '/board')

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        mean_iu = metrics['mean_iu']
        logger.info("Validation Results - Mean IU: {:.3f}".format(mean_iu))

    @evaluator.on(Events.EPOCH_STARTED)
    def plot_output(engine):
        model.eval()
        for i, batch in enumerate(val_loader):
            if i > 9:
                break
            val_x, val_y = batch
            val_x = val_x.to(device)
            with torch.no_grad():
                pred_y = model(val_x)

            orig_img, val_y = untransform(val_x.cpu().data[0], val_y[0])
            pred_y = pred_y.max(1)[1].cpu().data[0].numpy()
            pred_val = cm[pred_y]
            seg_val = cm[val_y]

            # matplotlib
            fig = plt.figure(figsize=(9, 3))
            plt.subplot(131)
            plt.imshow(orig_img)
            plt.axis("off")

            plt.subplot(132)
            plt.imshow(seg_val)
            plt.axis("off")

            plt.subplot(133)
            plt.imshow(pred_val)
            plt.axis("off")
            writer.add_figure('show_result', fig, i)

    evaluator.run(val_loader)
    writer.close()
