"""OFA supernet training."""

import os
import copy
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import xnas.core.config as config
from xnas.datasets.loader import get_normal_dataloader
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *
from xnas.logger.checkpoint import get_last_checkpoint

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# OFA
from xnas.runner.trainer import KDTrainer
from xnas.runner.scheduler import adjust_learning_rate_per_batch
from xnas.spaces.OFA.utils import init_model, list_mean, set_running_statistics


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)
# Upper dir for supernet
upper_dir = os.path.join(*cfg.OUT_DIR.split('/')[:-1]) 


SAMPLE_TIMES = 500


def main():
    setup_env()
    net = space_builder().cuda()
    # # [debug]
    # from xnas.spaces.OFA.MobileNetV3.ofa_cnn import _OFAMobileNetV3
    # net = _OFAMobileNetV3()
    checkpoint = torch.load("exp/search/OFA_trail_25/stage_ckpt/expand_2.pyth", map_location="cpu")
    net.load_state_dict(checkpoint["model_state"])
    
    logger.info("load finished.")
    
    [train_loader, valid_loader] = get_normal_dataloader()
    
    test_meter = meter.TestMeter(len(valid_loader))
    
    records = []
    best_err = 100.1
    
    for sample_time in range(SAMPLE_TIMES):
        netcfg = net.sample_active_subnet()
        set_running_statistics(net, valid_loader)
        net.eval()
        test_meter.reset(True)
        test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(valid_loader):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            preds = net(inputs)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            
            test_meter.iter_toc()
            test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
            # test_meter.log_iter_stats(0, cur_iter)
            test_meter.iter_tic()
        top1_err = test_meter.mb_top1_err.get_global_avg()
        top5_err = test_meter.mb_top5_err.get_global_avg()
        # return top1_err, top5_err
        
        records.append({
            'netcfg': netcfg,
            'top1err': top1_err,
            'top5err': top5_err,
        })
        
        if best_err > top1_err:
            best_err = top1_err
        logger.info("[{}/{}] top1_err:{} best_err:{}".format(sample_time+1, SAMPLE_TIMES, top1_err, best_err))
        with open(os.path.join(cfg.OUT_DIR, "sampled_info.pkl"), "wb") as f:
            pickle.dump(records, f)
    
    best = sorted(records, key=lambda d:d['top1err'])[0]
    logger.info("best architecture: {}".format(best))


if __name__ == '__main__':
    main()
