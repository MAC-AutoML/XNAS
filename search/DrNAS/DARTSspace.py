import os
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import xnas.core.config as config
import xnas.core.meters as meters
import xnas.core.logging as logging
import xnas.search_space.DrNAS.utils as utils
from xnas.search_algorithm.DrNAS import Architect

from xnas.core.builders import build_loss_fun, DrNAS_builder
from xnas.core.config import cfg
from xnas.core.timer import Timer
from xnas.core.trainer import setup_env, test_epoch
from xnas.datasets.loader import construct_loader


# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()
# Tensorboard supplement
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def main():

    setup_env()
    cudnn.benchmark = True  # DrNAS code sets this term to True.

    criterion = build_loss_fun().cuda()

    model = DrNAS_builder().cuda()
    architect = Architect(model, cfg)

    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
    )

    [train_loader, valid_loader] = construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE, cfg.SEARCH.DATAPATH
    )

    # configure progressive parameter
    epoch = 0
    ks = [6, 4]
    num_keeps = [7, 4]
    train_epochs = [2, 2] if "debug" in cfg.OUT_DIR else [25, 25]

    # lr_scheduler = lr_scheduler_builder(optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(sum(train_epochs)), eta_min=cfg.OPTIM.MIN_LR
    )

    train_meter = meters.TrainMeter(len(train_loader))
    val_meter = meters.TestMeter(len(valid_loader))

    # train_timer = Timer()
    for i, current_epoch in enumerate(train_epochs):
        logger.info("train period #{} total epochs {}".format(i, current_epoch))
        for e in range(current_epoch):
            lr = lr_scheduler.get_lr()[0]
            logger.info("epoch %d lr %e", epoch, lr)

            genotype = model.genotype()
            logger.info("genotype = %s", genotype)
            model.show_arch_parameters(logger)

            # train_timer.tic()
            # training
            top1err = train_epoch(
                train_loader,
                valid_loader,
                model,
                architect,
                criterion,
                optimizer,
                lr,
                train_meter,
                e,
            )
            logger.info("Top1 err:%f", top1err)
            # train_timer.toc()
            # print("epoch time:{}".format(train_timer.diff))

            # validation
            test_epoch(valid_loader, model, val_meter, epoch, writer)

            epoch += 1
            lr_scheduler.step()

            if epoch % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
                utils.save(
                    model, os.path.join(cfg.OUT_DIR, "weights_epo" + str(epoch) + ".pt")
                )

        # print("avg epoch time:{}".format(train_timer.average_time))
        # train_timer.reset()

        if not i == len(train_epochs) - 1:
            model.pruning(num_keeps[i + 1])
            # architect.pruning([model.mask_normal, model.mask_reduce])
            model.wider(ks[i + 1])
            optimizer = utils.configure_optimizer(
                optimizer,
                torch.optim.SGD(
                    model.parameters(),
                    cfg.OPTIM.BASE_LR,
                    momentum=cfg.OPTIM.MOMENTUM,
                    weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                ),
            )
            lr_scheduler = utils.configure_scheduler(
                lr_scheduler,
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(sum(train_epochs)), eta_min=cfg.OPTIM.MIN_LR
                ),
            )
            logger.info("pruning finish, %d ops left per edge", num_keeps[i + 1])
            logger.info("network wider finish, current pc parameter %d", ks[i + 1])

    genotype = model.genotype()
    logger.info("genotype = %s", genotype)
    model.show_arch_parameters(logger)


def train_epoch(
    train_loader,
    valid_loader,
    model,
    architect,
    criterion,
    optimizer,
    lr,
    train_meter,
    cur_epoch,
):
    train_meter.iter_tic()
    cur_step = cur_epoch * len(train_loader)
    writer.add_scalar("train/lr", lr, cur_step)

    valid_loader_iter = iter(valid_loader)

    for cur_iter, (trn_X, trn_y) in enumerate(train_loader):
        model.train()
        try:
            (val_X, val_y) = next(valid_loader_iter)
        except StopIteration:
            valid_loader_iter = iter(valid_loader)
            (val_X, val_y) = next(valid_loader_iter)
        # Transfer the data to the current GPU device
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda(non_blocking=True)
        val_X, val_y = val_X.cuda(), val_y.cuda(non_blocking=True)

        if cur_epoch >= 10:
            architect.step(
                trn_X, trn_y, val_X, val_y, lr, optimizer, unrolled=cfg.DARTS.UNROLLED
            )
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        logits = model(trn_X)
        loss = criterion(logits, trn_y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        top1_err, top5_err = meters.topk_errors(logits, trn_y, [1, 5])
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()

        # Update and log stats
        # TODO: multiply with NUM_GPUS are disabled before appling parallel
        # mb_size = trn_X.size(0) * cfg.NUM_GPUS
        mb_size = trn_X.size(0)
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        # write to tensorboard
        writer.add_scalar("train/loss", loss, cur_step)
        writer.add_scalar("train/top1_error", top1_err, cur_step)
        writer.add_scalar("train/top5_error", top5_err, cur_step)
        cur_step += 1
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return top1_err


if __name__ == "__main__":
    main()
    writer.close()
