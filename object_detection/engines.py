import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils import tensorboard
from ignite import engine
from ignite import metrics
from ignite import handlers
from ignite.contrib import handlers as contrib_handlers


def create_mask_rcnn_trainer(model: nn.Module, optimizer: optim.Optimizer, device=None, non_blocking: bool = False):
    if device:
        model.to(device)

    fn_prepare_batch = lambda batch: engine._prepare_batch(batch, device=device, non_blocking=non_blocking)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        image, targets = fn_prepare_batch(batch)
        losses = model(image, targets)

        loss = sum(loss for loss in losses.values())

        loss.backward()
        optimizer.step()

        losses = {k: v.item() for k, v in losses.items()}
        losses['loss'] = loss.item()
        return losses

    return engine.Engine(_update)


def create_mask_rcnn_evaluator(model: nn.Module, metrics, device=None, non_blocking: bool = False):
    if device:
        model.to(device)

    fn_prepare_batch = lambda batch: engine._prepare_batch(batch, device=device, non_blocking=non_blocking)

    def _update(engine, batch):
        # warning(will.brennan) - not putting model in eval mode because we want the losses!
        with torch.no_grad():
            image, targets = fn_prepare_batch(batch)
            losses = model(image, targets)

            losses = {k: v.item() for k, v in losses.items()}
            losses['loss'] = sum(losses.values())

        # note(will.brennan) - an ugly hack for metrics...
        return (losses, len(image))

    evaluator = engine.Engine(_update)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def attach_lr_scheduler(
    trainer: engine.Engine,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    writer: tensorboard.SummaryWriter,
):
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def update_lr(engine: engine.Engine):
        current_lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'epoch: {engine.state.epoch} - current lr: {current_lr}')
        writer.add_scalar('learning_rate', current_lr, engine.state.epoch)

        lr_scheduler.step()


def attach_training_logger(
    trainer: engine.Engine,
    writer: tensorboard.SummaryWriter,
    log_interval: int = 10,
):
    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(engine: engine.Engine):
        epoch_length = engine.state.epoch_length
        epoch = engine.state.epoch
        output = engine.state.output

        idx = engine.state.iteration
        idx_in_epoch = (engine.state.iteration - 1) % epoch_length + 1

        if idx_in_epoch % 10 != 0:
            return

        msg = ''
        for name, value in output.items():
            msg += f'{name}: {value:.4f} '
            writer.add_scalar(f'training/{name}', value, idx)
        logging.info(f'epoch[{epoch}] - iteration[{idx_in_epoch}/{epoch_length}] ' + msg)


def attach_metric_logger(
    trainer: engine.Engine,
    evaluator: engine.Engine,
    data_name: str,
    data_loader: data.DataLoader,
    writer: tensorboard.SummaryWriter,
):
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        evaluator.run(data_loader)

        def _to_message(metrics):
            message = ''

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    message += _to_message(metric_value)
                else:
                    writer.add_scalar(f'{data_name}/mean_{metric_name}', metric_value, engine.state.epoch)
                    message += f'{metric_name}: {metric_value:.3f} '

            return message

        message = _to_message(evaluator.state.metrics)
        logging.info(message)


def attach_model_checkpoint(trainer: engine.Engine, models: Dict[str, nn.Module]):
    def to_epoch(trainer: engine.Engine, event_name: str):
        return trainer.state.epoch

    handler = handlers.ModelCheckpoint(
        './models',
        'model',
        create_dir=True,
        require_empty=False,
        n_saved=None,
        global_step_transform=to_epoch,
    )
    trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, handler, models)
