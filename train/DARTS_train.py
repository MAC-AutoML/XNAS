"""train indepandent (can be augmented) model for DARTS"""
"""only support cifar10 now"""


import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import xnas.core.checkpoint as checkpoint
import xnas.core.config as config
import xnas.core.distributed as dist
import xnas.core.logging as logging
import xnas.core.meters as meters

from xnas.core.config import cfg
from xnas.core.builders import build_loss_fun, lr_scheduler_builder
from xnas.core.trainer import setup_env
from xnas.search_space.cellbased_DARTS_cnn import AugmentCNN
from xnas.datasets.loader import construct_loader

device = torch.device("cuda")

writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)

# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()


def main():
    setup_env()

    # 32 3 10 === 32 16 10
    # print(input_size, input_channels, n_classes, '===', cfg.SEARCH.IM_SIZE, cfg.SPACE.CHANNEL, cfg.SEARCH.NUM_CLASSES)

    loss_fun = build_loss_fun().cuda()
    use_aux = cfg.TRAIN.AUX_WEIGHT > 0.

    # SEARCH.INIT_CHANNEL as 3 for rgb and TRAIN.CHANNELS as 32 by manual.
    # IM_SIZE, CHANNEL and NUM_CLASSES should be same with search period.
    model = AugmentCNN(cfg.SEARCH.IM_SIZE, cfg.SEARCH.INPUT_CHANNEL, cfg.TRAIN.CHANNELS, 
                       cfg.SEARCH.NUM_CLASSES, cfg.TRAIN.LAYERS, use_aux, cfg.TRAIN.GENOTYPE)

    # TODO: Parallel
    # model = nn.DataParallel(model, device_ids=cfg.NUM_GPUS).to(device)
    model.cuda()

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), cfg.OPTIM.BASE_LR, momentum=cfg.OPTIM.MOMENTUM,
                                weight_decay=cfg.OPTIM.WEIGHT_DECAY)

    # Get data loader
    [train_loader, valid_loader] = construct_loader(
        cfg.TRAIN.DATASET, cfg.TRAIN.SPLIT, cfg.TRAIN.BATCH_SIZE)

    lr_scheduler = lr_scheduler_builder(optimizer)

    best_top1err = 0.

    # TODO: DALI backend support
    # if config.data_loader_type == 'DALI':
    #     len_train_loader = get_train_loader_len(config.dataset.lower(), config.batch_size, is_train=True)
    # else:
    len_train_loader = len(train_loader)

    # Training loop
    # TODO: RESUME

    train_meter = meters.TrainMeter(len(train_loader))
    valid_meter = meters.TestMeter(len(valid_loader))

    start_epoch = 0
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        
        drop_prob = cfg.TRAIN.DROP_PATH_PROB * cur_epoch / cfg.OPTIM.MAX_EPOCH
        if cfg.NUM_GPUS > 1:
            model.module.drop_path_prob(drop_prob)
        else:
            model.drop_path_prob(drop_prob)

        # Training
        train_epoch(train_loader, model, optimizer,
                    loss_fun, cur_epoch, train_meter)

        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(
                model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))

        lr_scheduler.step()

        # Validation
        cur_step = (cur_epoch + 1) * len(train_loader)
        top1_err = valid_epoch(valid_loader, model, loss_fun,
                           cur_epoch, cur_step, valid_meter)
        logger.info("top1 error@epoch {}: {}".format(cur_epoch + 1, top1_err))
        best_top1err = min(best_top1err, top1_err)

    logger.info("Final best Prec@1 = {:.4%}".format(100 - best_top1err))


def train_epoch(train_loader, model, optimizer, criterion, cur_epoch, train_meter):

    # TODO: DALI backend support
    # if config.data_loader_type == 'DALI':
    #     len_train_loader = get_train_loader_len(config.dataset.lower(), config.batch_size, is_train=True)
    # else:
    #     len_train_loader = len(train_loader)
    model.train()
    train_meter.iter_tic()
    cur_step = cur_epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('train/lr', cur_lr, cur_step)

    # TODO: DALI backend support
    # if config.data_loader_type == 'DALI':
    #     for cur_iter, data in enumerate(train_loader):
    #         X = data[0]["data"].cuda(non_blocking=True)
    #         y = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    #         if config.cutout_length > 0:
    #             X = cutout_batch(X, config.cutout_length)
    #         train_iter(X, y)
    #         cur_step += 1
    #     train_loader.reset()
    for cur_iter, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if cfg.TRAIN.AUX_WEIGHT > 0.:
            loss += cfg.TRAIN.AUX_WEIGHT * criterion(aux_logits, y)
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()

        top1_err, top5_err = meters.topk_errors(logits, y, [1, 5])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = X.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(top1_err, top5_err, loss, cur_lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        # write to tensorboard
        writer.add_scalar('train/loss', loss, cur_step)
        writer.add_scalar('train/top1_error', top1_err, cur_step)
        writer.add_scalar('train/top5_error', top5_err, cur_step)
        cur_step += 1
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def valid_epoch(valid_loader, model, criterion, cur_epoch, cur_step, valid_meter):
    model.eval()
    valid_meter.iter_tic()
    for cur_iter, (X, y) in enumerate(valid_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = model(X)
        loss = criterion(logits, y)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(logits, y, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        # NOTE: this line is disabled before.
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        valid_meter.iter_toc()
        # Update and log stats
        valid_meter.update_stats(
            top1_err, top5_err, X.size(0) * cfg.NUM_GPUS)
        valid_meter.log_iter_stats(cur_epoch, cur_iter)
        valid_meter.iter_tic()
    top1_err = valid_meter.mb_top1_err.get_win_median()
    valid_meter.log_epoch_stats(cur_epoch)
    valid_meter.reset()
    return top1_err

if __name__ == "__main__":
    main()
