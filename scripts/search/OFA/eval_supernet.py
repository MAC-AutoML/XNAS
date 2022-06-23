"""OFA supernet training."""

import os
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import xnas.core.config as config
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


def main():
    setup_env()
    torch.cuda.set_device(0)
    # device = torch.device("cuda", local_rank)
    # Network
    net = space_builder().to(0)
    init_model(net)
    # Loss function
    criterion = criterion_builder()
    # Data loaders
    [train_loader, valid_loader] = construct_loader()
    # Optimizers
    net_params = [
        # parameters with weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="exclude"), "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        # parameters without weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="include") , "weight_decay": 0}, 
    ]
    optimizer = optimizer_builder("SGD", net_params)
    # lr_scheduler = lr_scheduler_builder()     # OFA controls lr per iter & manually.
    
    ofa_teacher_model = None
    
    # build validation config
    validate_func_dict = {
        "image_size_list": {cfg.TEST.IM_SIZE},   # TODO: using multi-size test images.
        "ks_list": sorted({min(net.ks_list), max(net.ks_list)}),
        "expand_ratio_list": sorted({min(net.expand_ratio_list), max(net.expand_ratio_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
    }
    if cfg.OFA.TASK == 'normal':
        pass
    else:
        if cfg.OFA.TASK == 'kernel':
            validate_func_dict["ks_list"] = sorted(net.ks_list)
        elif cfg.OFA.TASK == 'depth':
            if (len(set(net.ks_list)) == 1) and (len(set(net.expand_ratio_list)) == 1):
                validate_func_dict["depth_list"] = net.depth_list.copy()
        elif cfg.OFA.TASK == 'expand':
            if len(set(net.ks_list)) == 1 and len(set(net.depth_list)) == 1:
                validate_func_dict["expand_ratio_list"] = net.expand_ratio_list.copy()
        else:
            raise NotImplementedError

    # Initialize Recorder
    ofa_trainer = OFATrainer(
        model=net,
        teacher_model=ofa_teacher_model,
        kd_ratio=cfg.OFA.KD_RATIO,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        train_loader=train_loader,
        test_loader=valid_loader,
    )
    # Resume
    # start_epoch = ofa_trainer.loading() if cfg.SEARCH.AUTO_RESUME else 0
        
    # # load last stage's checkpoint if not resume
    # if start_epoch == 0 and cfg.OFA.TASK != 'normal':
    #     load_last_stage_ckpt(cfg.OFA.TASK, cfg.OFA.PHASE)
    #     ofa_trainer.resume()    # only load the state_dict of model

    # cfg.SEARCH.WEIGHTS = '/home/xfey/XNAS/exp/search/OFA_trial_25/kernel_1/checkpoints/model_epoch_0110.pyth'
    cfg.SEARCH.WEIGHTS = '/home/xfey/XNAS/tests/weights/ofa_D4_E6_K357'
    ofa_trainer.resume()

    # Training
    logger.info("=== OFA | Task: {} | Phase: {} ===".format(cfg.OFA.TASK, cfg.OFA.PHASE))
    ofa_trainer.start()
    ofa_trainer.validate_all_subnets(1, 0, **validate_func_dict)
    ofa_trainer.finish()
    torch.cuda.empty_cache()


def load_last_stage_ckpt(task, phase):
    order = ['normal_1', 'kernel_1', 'depth_1', 'depth_2', 'expand_1', 'expand_2']
    cfg.SEARCH.WEIGHTS = os.path.join(
        upper_dir, 
        "stage_ckpt", 
        order[order.index('{}_{}'.format(task, phase)) - 1]+".pyth"
    )


class OFATrainer(KDTrainer):
    def __init__(self, model, teacher_model, kd_ratio, criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super().__init__(model, teacher_model, kd_ratio, criterion, optimizer, lr_scheduler, train_loader, test_loader)
    

    @torch.no_grad()
    def test_epoch(self, cur_epoch, rank=0):
        self.model.eval()
        self.test_meter.reset(True)
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            preds = self.model(inputs)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            
            self.test_meter.iter_toc()
            self.test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
            self.test_meter.log_iter_stats(cur_epoch, cur_iter)
            self.test_meter.iter_tic()
        top1_err = self.test_meter.mb_top1_err.get_win_avg()
        top5_err = self.test_meter.mb_top5_err.get_win_avg()
        # self.writer.add_scalar('val/top1_error', self.test_meter.mb_top1_err.get_win_avg(), cur_epoch)
        # self.writer.add_scalar('val/top5_error', self.test_meter.mb_top5_err.get_win_avg(), cur_epoch)
        # Log epoch stats
        self.test_meter.log_epoch_stats(cur_epoch)
        # self.test_meter.reset()
        return top1_err, top5_err


    def validate_all_subnets(
        self,
        cur_epoch,
        rank=0,
        image_size_list=None,
        ks_list=None,
        expand_ratio_list=None,
        depth_list=None,
        width_mult_list=None,
        additional_setting=None,
    ):
        # tmp_model = copy.deepcopy(self.model)
        # if isinstance(tmp_model, nn.parallel.DistributedDataParallel):
            # tmp_model = tmp_model.module
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model = self.model.module
        self.model.eval()
        
        assert image_size_list is not None, 'validate: image_size should not be None'
        if ks_list is None:
            ks_list = self.model.ks_list
        if expand_ratio_list is None:
            expand_ratio_list = self.model.expand_ratio_list
        if depth_list is None:
            depth_list = self.model.depth_list
        if width_mult_list is None:
            if "width_mult_list" in self.model.__dict__:
                width_mult_list = list(range(len(self.model.width_mult_list)))
            else:
                width_mult_list = [0]

        subnet_settings = []
        for d in depth_list:
            for e in expand_ratio_list:
                for k in ks_list:
                    for w in width_mult_list:
                        # Disable multi-size test images
                        # for img_size in image_size_list:
                        subnet_settings.append(
                            [
                                {
                                    # "image_size": img_size,
                                    "ks": k,
                                    "e": e,
                                    "d": d,
                                    "w": w,
                                },
                                # "R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),
                                "D%s-E%s-K%s-W%s" % (d, e, k, w),
                            ]
                        )
        if additional_setting is not None:
            subnet_settings += additional_setting
        logger.info("Validating {} subnets".format(len(subnet_settings)))
        
        top1errs, top5errs = [], []
        for i, (setting, name) in enumerate(subnet_settings):
            self.model.set_active_subnet(**setting)
            set_running_statistics(self.model, self.test_loader, rank)
            top1_err, top5_err = self.test_epoch(cur_epoch, rank=rank)
            logger.info("[{}/{}] subnet:{} | epoch:{} | top1:{} | top5:{}".format(i+1, len(subnet_settings), name, cur_epoch+1, top1_err, top5_err))
            top1errs.append(top1_err)
            top5errs.append(top5_err)
        
        # self.writer.add_scalar('val/top1_error', list_mean(top1errs), cur_epoch)
        # self.writer.add_scalar('val/top5_error', list_mean(top5errs), cur_epoch)
        logger.info("Average@all_subnets top1_err:{} top5_err:{}".format(list_mean(top1errs), list_mean(top5errs)))
            
        # Saving best model
        if self.best_err > top1_err:
            self.best_err = top1_err
            self.saving(cur_epoch, best=True)


if __name__ == '__main__':
    main()
