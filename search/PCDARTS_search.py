import gc
import os
import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.tensorboard import SummaryWriter

import xnas.core.benchmark as benchmark
import xnas.core.checkpoint as checkpoint
import xnas.core.config as config
import xnas.core.logging as logging
import xnas.core.meters as meters
from xnas.core.timer import Timer
from xnas.core.builders import build_space, build_loss_fun, lr_scheduler_builder
from xnas.core.config import cfg
from xnas.core.trainer import setup_env, test_epoch
from xnas.datasets.cifar10 import data_transforms_cifar10
from xnas.search_algorithm.PCDARTS import *
from DARTS_search import darts_load_checkpoint, darts_save_checkpoint


# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()

# Tensorboard supplement
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def pcdarts_train_model():
    """train PC-DARTS model"""
    setup_env()
    # Loading search space
    search_space = build_space()
    # TODO: fix the complexity function
    # search_space = setup_model()
    # Init controller and architect
    loss_fun = build_loss_fun().cuda()
    pcdarts_controller = PCDartsCNNController(search_space, loss_fun)
    pcdarts_controller.cuda()
    architect = Architect(
        pcdarts_controller, cfg.OPTIM.MOMENTUM, cfg.OPTIM.WEIGHT_DECAY)

    # Load dataset
    train_transform, valid_transform = data_transforms_cifar10(cutout_length=0)

    train_data = dset.CIFAR10(
        root=cfg.SEARCH.DATASET, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(cfg.SEARCH.SPLIT[0] * num_train))

    train_ = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.SEARCH.BATCH_SIZE,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
    val_ = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.SEARCH.BATCH_SIZE,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=True, num_workers=2)

    # weights optimizer
    w_optim = torch.optim.SGD(pcdarts_controller.weights(),
                              cfg.OPTIM.BASE_LR,
                              momentum=cfg.OPTIM.MOMENTUM,
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    # alphas optimizer
    a_optim = torch.optim.Adam(pcdarts_controller.alphas(),
                               cfg.DARTS.ALPHA_LR,
                               betas=(0.5, 0.999),
                               weight_decay=cfg.DARTS.ALPHA_WEIGHT_DECAY)
    lr_scheduler = lr_scheduler_builder(w_optim)
    # Init meters
    train_meter = meters.TrainMeter(len(train_))
    val_meter = meters.TestMeter(len(val_))
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.SEARCH.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = darts_load_checkpoint(
            last_checkpoint, pcdarts_controller, w_optim, a_optim)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.SEARCH.WEIGHTS:
        darts_load_checkpoint(cfg.SEARCH.WEIGHTS, pcdarts_controller)
        logger.info("Loaded initial weights from: {}".format(
            cfg.SEARCH.WEIGHTS))
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(pcdarts_controller, loss_fun, train_, val_)
    # Setup timer
    train_timer = Timer()
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    train_timer.tic()
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        lr = lr_scheduler.get_last_lr()[0]
        train_epoch(train_, val_, pcdarts_controller, architect,
                    loss_fun, w_optim, a_optim, lr, train_meter, cur_epoch)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
            checkpoint_file = darts_save_checkpoint(
                pcdarts_controller, w_optim, a_optim, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        lr_scheduler.step()
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.SEARCH.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            logger.info("Start testing")
            test_epoch(val_, pcdarts_controller, val_meter, cur_epoch, writer)
            logger.info(
                "###############Optimal genotype at epoch: {}############".format(cur_epoch))
            logger.info(pcdarts_controller.genotype())
            logger.info(
                "########################################################")
            pcdarts_controller.print_alphas(logger)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
        gc.collect()
    train_timer.toc()
    logger.info("Overall training time (hr) is:{}".format(
        str(train_timer.total_time)))


def train_epoch(train_loader, valid_loader, model, architect, loss_fun, w_optimizer, alpha_optimizer, lr, train_meter, cur_epoch):
    model.train()
    train_meter.iter_tic()
    cur_step = cur_epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)
    # scale the grad in amp, amp only support the newest version
    scaler = torch.cuda.amp.GradScaler() if cfg.SEARCH.AMP & hasattr(
        torch.cuda.amp, 'autocast') else None
    valid_loader_iter = iter(valid_loader)
    for cur_iter, (trn_X, trn_y) in enumerate(train_loader):
        try:
            (val_X, val_y) = next(valid_loader_iter)
        except StopIteration:
            valid_loader_iter = iter(valid_loader)
            (val_X, val_y) = next(valid_loader_iter)
        # Transfer the data to the current GPU device
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda(non_blocking=True)
        val_X, val_y = val_X.cuda(), val_y.cuda(non_blocking=True)
        # phase 2. architect step (alpha)
        if cur_epoch >= 15:
            if cfg.OPTIM.UNROLLED == False:
                alpha_optimizer.zero_grad()
                aloss = architect.net.loss(val_X, val_y)
                aloss.backward()
                alpha_optimizer.step()
            else:
                alpha_optimizer.zero_grad()
                architect.unrolled_backward(
                    trn_X, trn_y, val_X, val_y, lr, w_optimizer, unrolled=cfg.DARTS.SECOND)
                alpha_optimizer.step()

        # phase 1. child network step (w)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Perform the forward pass in AMP
                preds = model(trn_X)
                # Compute the loss in AMP
                loss = loss_fun(preds, trn_y)
                # Perform the backward pass in AMP
                w_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(w_optimizer)
                # Updates the scale for next iteration.
                scaler.update()
        else:
            preds = model(trn_X)
            # Compute the loss
            loss = loss_fun(preds, trn_y)
            # Perform the backward pass
            w_optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.weights(), cfg.OPTIM.GRAD_CLIP)
            # Update the parameters
            w_optimizer.step()
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, trn_y, [1, 5])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = trn_X.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
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


if __name__ == "__main__":
    pcdarts_train_model()
    writer.close()
