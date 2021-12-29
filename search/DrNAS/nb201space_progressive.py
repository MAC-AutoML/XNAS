import os
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

import xnas.core.logging as logging
import xnas.core.config as config
import xnas.core.meters as meters
import xnas.search_space.DrNAS.utils as utils
from xnas.core.builders import build_loss_fun, DrNAS_builder
from xnas.core.config import cfg
from xnas.core.timer import Timer
from xnas.core.trainer import setup_env, test_epoch
from xnas.datasets.loader import construct_loader
from xnas.search_algorithm.DrNAS import Architect

from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API


# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()
# Tensorboard supplement
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def distill(result):
    result = result.split("\n")
    cifar10 = result[5].replace(" ", "").split(":")
    cifar100 = result[7].replace(" ", "").split(":")
    imagenet16 = result[9].replace(" ", "").split(":")

    cifar10_train = float(cifar10[1].strip(",test")[-7:-2].strip("="))
    cifar10_test = float(cifar10[2][-7:-2].strip("="))
    cifar100_train = float(cifar100[1].strip(",valid")[-7:-2].strip("="))
    cifar100_valid = float(cifar100[2].strip(",test")[-7:-2].strip("="))
    cifar100_test = float(cifar100[3][-7:-2].strip("="))
    imagenet16_train = float(imagenet16[1].strip(",valid")[-7:-2].strip("="))
    imagenet16_valid = float(imagenet16[2].strip(",test")[-7:-2].strip("="))
    imagenet16_test = float(imagenet16[3][-7:-2].strip("="))

    return (
        cifar10_train,
        cifar10_test,
        cifar100_train,
        cifar100_valid,
        cifar100_test,
        imagenet16_train,
        imagenet16_valid,
        imagenet16_test,
    )


def main():

    setup_env()
    # follow DrNAS settings.
    torch.set_num_threads(3)
    cudnn.benchmark = True

    if not "debug" in cfg.OUT_DIR:
        api = API("./data/NAS-Bench-201-v1_1-096897.pth")

    criterion = build_loss_fun().cuda()

    assert cfg.DRNAS.METHOD in ["snas", "dirichlet", "darts"], "method not supported."

    if cfg.DRNAS.METHOD == "snas":
        # Create the decrease step for the gumbel softmax temperature
        # cfg.OPTIM.MAX_EPOCH = 100
        [tau_min, tau_max] = cfg.DRNAS.TAU
        # Create the decrease step for the gumbel softmax temperature
        tau_step = (tau_min - tau_max) / cfg.OPTIM.MAX_EPOCH
        tau_epoch = tau_max

    model = DrNAS_builder().cuda()

    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.get_weights(),
        cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
    )

    train_loader, valid_loader = construct_loader(
        cfg.SEARCH.DATASET,
        cfg.SEARCH.SPLIT,
        cfg.SEARCH.BATCH_SIZE,
        cfg.SEARCH.DATAPATH,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )

    architect = Architect(model, cfg)

    # configure progressive parameter
    epoch = 0
    ks = [4, 2]
    num_keeps = [5, 3]
    train_epochs = [2, 2] if "debug" in cfg.OUT_DIR else [50, 50]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(sum(train_epochs)), eta_min=cfg.OPTIM.MIN_LR
    )

    train_meter = meters.TrainMeter(len(train_loader))
    val_meter = meters.TestMeter(len(valid_loader))

    # train_timer = Timer()
    for i, current_epochs in enumerate(train_epochs):
        for e in range(current_epochs):
            lr = scheduler.get_lr()[0]
            logger.info("epoch %d lr %e", epoch, lr)
            genotype = model.genotype()
            logger.info("genotype = %s", genotype)
            model.show_arch_parameters(logger)

            # training
            # train_timer.tic()
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

            if not "debug" in cfg.OUT_DIR:
                # nasbench201
                result = api.query_by_arch(model.genotype())
                logger.info("{:}".format(result))
                (
                    cifar10_train,
                    cifar10_test,
                    cifar100_train,
                    cifar100_valid,
                    cifar100_test,
                    imagenet16_train,
                    imagenet16_valid,
                    imagenet16_test,
                ) = distill(result)
                logger.info("cifar10 train %f test %f", cifar10_train, cifar10_test)
                logger.info(
                    "cifar100 train %f valid %f test %f",
                    cifar100_train,
                    cifar100_valid,
                    cifar100_test,
                )
                logger.info(
                    "imagenet16 train %f valid %f test %f",
                    imagenet16_train,
                    imagenet16_valid,
                    imagenet16_test,
                )

                # tensorboard
                writer.add_scalars(
                    "nasbench201/cifar10",
                    {"train": cifar10_train, "test": cifar10_test},
                    epoch,
                )
                writer.add_scalars(
                    "nasbench201/cifar100",
                    {
                        "train": cifar100_train,
                        "valid": cifar100_valid,
                        "test": cifar100_test,
                    },
                    epoch,
                )
                writer.add_scalars(
                    "nasbench201/imagenet16",
                    {
                        "train": imagenet16_train,
                        "valid": imagenet16_valid,
                        "test": imagenet16_test,
                    },
                    epoch,
                )

                utils.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "alpha": model.arch_parameters(),
                    },
                    False,
                    cfg.OUT_DIR,
                )

            epoch += 1
            scheduler.step()
            if cfg.DRNAS.METHOD == "snas":
                # Decrease the temperature for the gumbel softmax linearly
                tau_epoch += tau_step
                logger.info("tau %f", tau_epoch)
                model.set_tau(tau_epoch)

        if not i == len(train_epochs) - 1:
            model.pruning(num_keeps[i + 1])
            # architect.pruning([model._mask])
            model.wider(ks[i + 1])
            optimizer = utils.configure_optimizer(
                optimizer,
                torch.optim.SGD(
                    model.get_weights(),
                    cfg.OPTIM.BASE_LR,
                    momentum=cfg.OPTIM.MOMENTUM,
                    weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                ),
            )
            scheduler = utils.configure_scheduler(
                scheduler,
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(sum(train_epochs)), eta_min=cfg.OPTIM.MIN_LR
                ),
            )
            logger.info("pruning finish, %d ops left per edge", num_keeps[i + 1])
            logger.info("network wider finish, current pc parameter %d", ks[i + 1])

    genotype = model.genotype()
    logger.info("genotype = %s", genotype)
    model.show_arch_parameters(logger)
    writer.close()


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
                trn_X, trn_y, val_X, val_y, lr, optimizer, unrolled=cfg.DRNAS.UNROLLED
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
