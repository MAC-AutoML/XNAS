""" Search cell """
import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

import xnas.core.checkpoint as checkpoint
import xnas.core.config as config
import xnas.core.logging as logging
import xnas.core.meters as meters
from xnas.core.builders import build_space, lr_scheduler_builder, sng_builder
from xnas.core.config import cfg
from xnas.core.trainer import setup_env
from xnas.core.utils import index_to_one_hot
from xnas.datasets.loader import _construct_loader

# config load and assert
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()
# tensorboard
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def main():
    setup_env()
    # loadiong search space
    search_space = build_space()
    search_space.cuda()
    # init controller and architect
    loss_fun = nn.CrossEntropyLoss().cuda()

    # weights optimizer
    w_optim = torch.optim.SGD(search_space.parameters(), cfg.OPTIM.BASE_LR, momentum=cfg.OPTIM.MOMENTUM,
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    # load dataset
    [train_, val_] = _construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)

    distribution_optimizer = sng_builder([search_space.num_ops]*search_space.all_edges)
    lr_scheduler = lr_scheduler_builder(w_optim)
    num_ops, total_edges = search_space.num_ops, search_space.all_edges
    # training loop
    logger.info("start warm up training")
    warm_train_meter = meters.TrainMeter(len(train_))
    warm_val_meter = meters.TestMeter(len(val_))
    start_time = time.time()
    _over_all_epoch = 0
    for epoch in range(cfg.OPTIM.WARMUP_EPOCHS):
        # lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]
        # warm up training
        array_sample = [random.sample(list(range(num_ops)), num_ops) for i in range(total_edges)]
        array_sample = np.array(array_sample)
        for i in range(num_ops):
            sample = np.transpose(array_sample[:, i])
            sample = index_to_one_hot(sample, distribution_optimizer.p_model.Cmax)
            _over_all_epoch += 1
            train(train_, val_, search_space, w_optim, lr, _over_all_epoch, sample, loss_fun, warm_train_meter)
            top1 = test_epoch(val_, search_space, warm_val_meter, _over_all_epoch, sample, writer)
    logger.info("end warm up training")
    logger.info("start One shot searching")
    train_meter = meters.TrainMeter(len(train_))
    val_meter = meters.TestMeter(len(val_))
    for epoch in range(cfg.OPTIM.MAX_EPOCH):
        if hasattr(distribution_optimizer, 'training_finish'):
            if distribution_optimizer.training_finish:
                break
        lr = w_optim.param_groups[0]['lr']
        sample = distribution_optimizer.sampling()

        # training
        _over_all_epoch += 1
        train(train_, val_, search_space, w_optim, lr, _over_all_epoch, sample, loss_fun, train_meter)

        # validation
        top1 = test_epoch(val_, search_space, val_meter, _over_all_epoch, sample, writer)
        lr_scheduler.step()
        distribution_optimizer.record_information(sample, top1)
        distribution_optimizer.update()

        # Evaluate the model
        next_epoch = epoch + 1
        if next_epoch % cfg.SEARCH.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            logger.info("Start testing")
            logger.info("###############Optimal genotype at epoch: {}############".format(epoch))
            logger.info(search_space.genotype(distribution_optimizer.p_model.theta))
            logger.info("########################################################")
            logger.info("####### ALPHA #######")
            for alpha in distribution_optimizer.p_model.theta:
                logger.info(alpha)
            logger.info("#####################")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
        gc.collect()
    for epoch in range(cfg.OPTIM.FINAL_EPOCH):
        sample = distribution_optimizer.sampling_best()
        _over_all_epoch += 1
        train(train_, val_, search_space, w_optim, lr, _over_all_epoch, sample, loss_fun, train_meter)
        test_epoch(val_, search_space, val_meter, _over_all_epoch, sample, writer)
    end_time = time.time()
    logger.info("Overall training time (hr) is:{}".format(str((end_time-start_time)/3600.)))


def train(train_loader, valid_loader, model, w_optim, lr, epoch, sample, net_crit, train_meter):

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    train_meter.iter_tic()
    scaler = torch.cuda.amp.GradScaler() if cfg.SEARCH.AMP & hasattr(
        torch.cuda.amp, 'autocast') else None

    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda()
        # phase 1. child network step (w)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Perform the forward pass in AMP
                preds = model(trn_X, sample)
                # Compute the loss in AMP
                loss = net_crit(preds, trn_y)
                # Perform the backward pass in AMP
                w_optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(w_optim)
                # Updates the scale for next iteration.
                scaler.update()
        else:
            preds = model(trn_X, sample)
            # Compute the loss
            loss = net_crit(preds, trn_y)
            # Perform the backward pass
            w_optim.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
            # Update the parameters
            w_optim.step()

        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, trn_y, [1, 5])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = trn_X.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(epoch, step)
        train_meter.iter_tic()
        # write to tensorboard
        writer.add_scalar('train/loss', loss, cur_step)
        writer.add_scalar('train/top1_error', top1_err, cur_step)
        writer.add_scalar('train/top5_error', top5_err, cur_step)
        cur_step += 1

    # Log epoch stats
    train_meter.log_epoch_stats(epoch)
    train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch, sample, tensorboard_writer=None,):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # using AMP
        if cfg.SEARCH.AMP & hasattr(torch.cuda.amp, 'autocast'):
            with torch.cuda.amp.autocast():
                # Compute the predictions
                preds = model(inputs, sample)
        else:
            # Compute the predictions
            preds = model(inputs, sample)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        # top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    top1_err = test_meter.mb_top1_err.get_win_median()
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar(
            'val/top1_error', test_meter.mb_top1_err.get_win_median(), cur_epoch)
        tensorboard_writer.add_scalar(
            'val/top5_error', test_meter.mb_top5_err.get_win_median(), cur_epoch)
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()
    return top1_err


if __name__ == "__main__":
    main()
