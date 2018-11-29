# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss, RunningAverage
from tensorboardX import SummaryWriter

from data.transforms import build_untransform
from data.transforms.transforms import COLORMAP
from utils.metric import Label_Accuracy

plt.switch_backend('agg')


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn
):
    cm = np.array(COLORMAP).astype(np.uint8)
    untransform = build_untransform(cfg)

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS
    device = cfg.MODEL.DEVICE
    output_dir = cfg.OUTPUT_DIR

    logger = logging.getLogger("FCN_Model.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'mean_iu': Label_Accuracy(cfg.MODEL.NUM_CLASSES),
                                                            'loss': Loss(loss_fn)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, 'fcn', checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)
    writer = SummaryWriter(output_dir + '/board')

    # automatically adding handlers via a special `attach` method of `RunningAverage` handler
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    # automatically adding handlers via a special `attach` method of `Checkpointer` handler
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})

    # automatically adding handlers via a special `attach` method of `Timer` handler
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))
            writer.add_scalars("loss", {'train': engine.state.metrics['avg_loss']}, engine.state.iteration)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        mean_iu = metrics['mean_iu']
        avg_loss = metrics['loss']
        logger.info("Training Results - Epoch: {} Mean IU: {:.3f} Avg Loss: {:.3f}"
                    .format(engine.state.epoch, mean_iu, avg_loss))
        writer.add_scalars("mean_iu", {'train': mean_iu}, engine.state.epoch)

    if val_loader is not None:
        # adding handlers using `trainer.on` decorator API
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            mean_iu = metrics['mean_iu']
            avg_loss = metrics['loss']
            logger.info("Validation Results - Epoch: {} Mean IU: {:.3f} Avg Loss: {:.3f}"
                        .format(engine.state.epoch, mean_iu, avg_loss)
                        )
            writer.add_scalars("loss", {'validation': avg_loss}, engine.state.iteration)
            writer.add_scalars("mean_iu", {'validation': mean_iu}, engine.state.epoch)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def plot_output(engine):
        model.eval()
        dataset = val_loader.dataset
        idx = np.random.choice(np.arange(len(dataset)), size=1).item()
        val_x, val_y = dataset[idx]
        val_x = val_x.to(device)
        with torch.no_grad():
            pred_y = model(val_x.unsqueeze(0))

        orig_img, val_y = untransform(val_x.cpu().data, val_y)
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
        writer.add_figure('show_result', fig, engine.state.iteration)

    trainer.run(train_loader, max_epochs=epochs)
    writer.close()
