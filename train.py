import argparse
import logging

from torch import nn
from torch import optim
from torch.utils import data
from ignite import engine
from torch.utils import tensorboard

from object_detection import LabelMeDataset
from object_detection import create_data_loaders
from object_detection import MaskRCNN
from object_detection import create_mask_rcnn_trainer
from object_detection import create_mask_rcnn_evaluator
from object_detection import LossAverager

from object_detection import attach_lr_scheduler
from object_detection import attach_training_logger
from object_detection import attach_model_checkpoint
from object_detection import attach_metric_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--model-tag', type=str, required=True)

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--initial-lr', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    logging.info('creating dataset and data loaders')

    # assert args.train != args.val
    train_dataset = LabelMeDataset(args.train, use_augmentation=True)
    val_dataset = LabelMeDataset(args.val, use_augmentation=False)
    assert train_dataset.categories == val_dataset.categories

    train_loader, train_metrics_loader, val_metrics_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    logging.info(f'creating model and optimizer with initial lr of {args.initial_lr}')
    model = MaskRCNN(train_dataset.categories)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.RMSprop(params=[p for p in model.parameters() if p.requires_grad], lr=args.initial_lr)

    logging.info('creating trainer and evaluator engines')
    trainer = create_mask_rcnn_trainer(model=model, optimizer=optimizer, device='cuda', non_blocking=True)
    # note(will.brennan) - our evaluator just reports losses! we only want to see if its overfitting!
    evaluator = create_mask_rcnn_evaluator(
        model, metrics={
            'losses': LossAverager(),
        }, device='cuda', non_blocking=True
    )

    logging.info(f'creating summary writer with tag {args.model_tag}')
    writer = tensorboard.SummaryWriter(log_dir=f'logs/{args.model_tag}')

    logging.info('attaching lr scheduler')
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    attach_lr_scheduler(trainer, lr_scheduler, writer)

    logging.info('attaching event driven calls')
    attach_model_checkpoint(trainer, {args.model_tag: model.module})
    attach_training_logger(trainer, writer=writer)

    attach_metric_logger(trainer, evaluator, 'train', train_metrics_loader, writer)
    attach_metric_logger(trainer, evaluator, 'val', val_metrics_loader, writer)

    logging.info('training...')
    trainer.run(train_loader, max_epochs=args.num_epochs)
