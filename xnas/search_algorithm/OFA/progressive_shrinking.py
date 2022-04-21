from tracemalloc import start
import torch.nn as nn
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from xnas.core.config import cfg
import xnas.core.meters as meters
from xnas.core.timer import Timer
from xnas.core.trainer import test_epoch
import xnas.core.checkpoint as checkpoint
from xnas.core.builders import lr_scheduler_builder, warmup_scheduler_builder
from xnas.search_space.OFA.utils import (
    DistributedMetric,
    list_mean,
    subset_mean,
)

__all__ = [
    "validate",
    "train_one_epoch",
    "train",
]


def validate(
    val_, net, val_meter,
    cur_epoch, logger,
    
    image_size_list=None,
    ks_list=None,
    expand_ratio_list=None,
    depth_list=None,
    width_mult_list=None,
    additional_setting=None,
):
    # net
    dynamic_net = net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module
    # eval mode
    dynamic_net.eval()

    # net config
    assert image_size_list is not None, 'validate: image_size should not be None'

    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list
    if width_mult_list is None:
        if "width_mult_list" in dynamic_net.__dict__:
            width_mult_list = list(range(len(dynamic_net.width_mult_list)))
        else:
            width_mult_list = [0]


    # 获取所有subnet的setting
    subnet_settings = []
    img_size = cfg.TEST.IM_SIZE
    for d in depth_list:
        for e in expand_ratio_list:
            for k in ks_list:
                for w in width_mult_list:
                    for img_size in image_size_list:
                        subnet_settings.append(
                            [
                                {
                                    "img_size": img_size,
                                    "d": d,
                                    "e": e,
                                    "ks": k,
                                    "w": w,
                                },
                                "R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),
                            ]
                        )
    if additional_setting is not None:
        subnet_settings += additional_setting


    # 遍历评估所有subnet
    for setting, name in subnet_settings:
        dynamic_net.set_active_subnet(**setting)
        logger.info('epoch: '+str(cur_epoch+1) + '  ||  validate subnet: '+ name)
        test_epoch(val_, dynamic_net, val_meter, cur_epoch)


def train_one_epoch(
        train_, net, train_criterion,
        optimizer, lr_scheduler, writer, train_meter,
        cur_epoch,
    ):
    
    nBatch = len(train_)
    cur_step = cur_epoch*nBatch

    for cur_iter, (images, labels) in enumerate(train_):
        cur_step += 1
        net.train()
        train_meter.iter_tic() # 初始化时间

        images, labels = images.cuda(), labels.cuda()

        # clean gradients
        net.zero_grad()

        # set random seed before sampling
        subnet_seed = int("%d%.3d" % (cur_step, 0))
        random.seed(subnet_seed)
        # subset setting
        subnet_settings = net.sample_active_subnet()
        subnet_str = ",".join(
            [
                "%s_%s"
                % (
                    key,
                    "%.1f" % subset_mean(val, 0)
                    if isinstance(val, list)
                    else val,
                )
                for key, val in subnet_settings.items()
            ]
        )
        # compute output
        output = net(images)
        loss = train_criterion(output, labels)
        loss.backward()
        optimizer.step()

        # measure top1&top5 error
        top1_err, top5_err = meters.topk_errors(output, labels, [1, 5])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = images.size(0)
        cur_lr = lr_scheduler.get_last_lr()[0]
        train_meter.update_stats(top1_err, top5_err, loss, cur_lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        # write to tensorboard
        writer.add_scalar('train/lr', cur_lr, cur_step)
        writer.add_scalar('train/loss', loss, cur_step)
        writer.add_scalar('train/top1_error', top1_err, cur_step)
        writer.add_scalar('train/top5_error', top5_err, cur_step)
        # update lr
        lr_scheduler.step(cur_epoch + cur_iter / nBatch)
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return


def train(
    net, start_epoch, optimizer, train_criterion, train_, val_,
    logger, writer, validate_func_dict,
    ):
    scheduler = warmup_scheduler_builder(
        optimizer,
        lr_scheduler_builder(optimizer, last_epoch=max(start_epoch-cfg.OPTIM.WARMUP_EPOCHS-1, 0)-1),
        last_epoch=start_epoch-1
    )
    
    ### init meters
    train_timer = Timer()
    train_meter = meters.TrainMeter(len(train_), include_lr_warmup=True)
    val_meter = meters.TestMeter(len(val_), include_lr_warmup=True)
    logger.info("Start epoch: {}".format(start_epoch + 1))
    train_timer.tic()
    for cur_epoch in range(start_epoch, cfg.OPTIM.WARMUP_EPOCHS+cfg.OPTIM.MAX_EPOCH):
        train_one_epoch(
            train_, net, train_criterion,
            optimizer, scheduler, writer, train_meter,
            cur_epoch,
        )
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(net, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.SEARCH.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            logger.info("Start testing")
            validate(
                val_, net, val_meter,
                cur_epoch, logger,
                **validate_func_dict
            )
        # TODO：OFA支持保存validation accuracy最高的checkpoint
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    train_timer.toc()
    logger.info("Overall training time (hr) is:{}".format(str(train_timer.total_time)))
