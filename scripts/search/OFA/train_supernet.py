"""OFA supernet training."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import xnas.core.config as config
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# OFA
from xnas.runner.trainer import KDTrainer
from xnas.runner.scheduler import adjust_learning_rate_per_batch
from xnas.spaces.OFA.utils import init_model, list_mean


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

def main():
    setup_env()
    # Network
    net = space_builder().cuda()
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
    
    # [debug]
    from xnas.spaces.OFA.MobileNetV3.ofa_cnn import _OFAMobileNetV3
    net = _OFAMobileNetV3()
    
    if cfg.OFA.KD_RATIO > 0.:
        logger.info("Using knowledge distillation with KD_ratio={}".format(cfg.OFA.KD_RATIO))
        from xnas.spaces.OFA.MobileNetV3.cnn import MobileNetV3Large
        ofa_teacher_model = MobileNetV3Large(
            n_classes=cfg.LOADER.NUM_CLASSES,
            dropout_rate=0.,
            width_mult=1.0,
            ks=7,
            expand_ratio=6,
            depth_param=4,
        ).cuda()
        ofa_teacher_model.load_state_dict(torch.load(cfg.OFA.KD_PATH)["state_dict"])
    else:
        ofa_teacher_model = None

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
    start_epoch = ofa_trainer.loading() if cfg.SEARCH.AUTO_RESUME else 0
    
    # build validation config
    validate_func_dict = {
        "image_size_list": {cfg.TEST.IM_SIZE},   # TODO: using multi-size test images.
        "ks_list": sorted({min(net.ks_list), max(net.ks_list)}),
        "expand_ratio_list": sorted({min(net.expand_ratio_list), max(net.expand_ratio_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
    }
    if cfg.OFA.TASK == 'normal':
        pass
    elif cfg.OFA.TASK == 'kernel':
        validate_func_dict["ks_list"] = sorted(net.ks_list)
    elif cfg.OFA.TASK == 'depth':
        if (len(set(net.ks_list)) == 1) and (len(set(net.expand_ratio_list)) == 1):
            validate_func_dict["depth_list"] = net.depth_list.copy()
    elif cfg.OFA.TASK == 'expand':
        if len(set(net.ks_list)) == 1 and len(set(net.depth_list)) == 1:
            validate_func_dict["expand_ratio_list"] = net.expand_ratio_list.copy()
    else:
        raise NotImplementedError

    # Training
    logger.info("=== OFA | Task: {} | Phase: {} ===".format(cfg.OFA.TASK, cfg.OFA.PHASE))
    ofa_trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.WARMUP_EPOCH+cfg.OPTIM.MAX_EPOCH):
        ofa_trainer.train_epoch(cur_epoch)
        if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
            ofa_trainer.validate_all_subnets(cur_epoch, **validate_func_dict)
    ofa_trainer.finish()


class OFATrainer(KDTrainer):
    def __init__(self, model, teacher_model, kd_ratio, criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super().__init__(model, teacher_model, kd_ratio, criterion, optimizer, lr_scheduler, train_loader, test_loader)
    
    def train_epoch(self, cur_epoch):
        self.model.train()
        # lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        # self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            
            # Adjust lr per iter
            cur_lr = adjust_learning_rate_per_batch(
                epoch=cur_epoch,
                n_iter=len(self.train_loader),
                iter=cur_iter,
                warmup=(cur_epoch < cfg.OPTIM.WARMUP_EPOCH),
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = cur_lr
            self.writer.add_scalar('train/lr', cur_lr, cur_step)
            
            # Knowledge Distillation from teacher model
            if (self.teacher_model is not None) and (self.kd_ratio > 0.):
                self.teacher_model.train()
                with torch.no_grad():
                    soft_logits = self.teacher_model(inputs).detach()
                    soft_label = F.softmax(soft_logits, dim=1)
                    kd_loss = self.celoss_st(preds, soft_label)

            # [Progressive Shrinking] sampling subnets and training.
            self.model.zero_grad()
            losses, top1errs, top5errs = [], [], []
            for i_subnet in range(cfg.OFA.SUBNET_BATCH_SIZE):
                subnet_seed = int("%d%.3d%.3d" % (cur_step, i_subnet, 0))
                random.seed(subnet_seed)
                subnet_settings = self.model.sample_active_subnet()
                logger.info("epoch:{} iter:{} subnet_settings:{}".format(cur_epoch, cur_iter, subnet_settings))
                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                if (self.teacher_model is not None) and (self.kd_ratio > 0.):
                    loss += kd_loss * self.kd_ratio
                loss.backward()
                
                # Record
                losses.append(loss.item())
                top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
                top1errs.append(top1_err.item()), top5errs.append(top5_err.item())
                
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()   # already called model.zero_grad()

            # Compute the errors of current iteration
            i_loss, i_top1err, i_top5err = list_mean(losses), list_mean(top1errs), list_mean(top5errs)
            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(i_top1err, i_top5err, i_loss, cur_lr, inputs.size(0))
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalar('train/loss', i_loss, cur_step)
            self.writer.add_scalar('train/top1_error', i_top1err, cur_step)
            self.writer.add_scalar('train/top5_error', i_top5err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        # self.lr_scheduler.step()
        # Saving checkpoint
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)


    @torch.no_grad()
    def test_epoch(self, cur_epoch):
        self.model.eval()
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            preds = self.model(inputs)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            
            self.test_meter.iter_toc()
            self.test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
            self.test_meter.log_iter_stats(cur_epoch, cur_iter)
            self.test_meter.iter_tic()
        top1_err = self.test_meter.mb_top1_err.get_win_median()
        top5_err = self.test_meter.mb_top5_err.get_win_median()
        # self.writer.add_scalar('val/top1_error', self.test_meter.mb_top1_err.get_win_median(), cur_epoch)
        # self.writer.add_scalar('val/top5_error', self.test_meter.mb_top5_err.get_win_median(), cur_epoch)
        # Log epoch stats
        self.test_meter.log_epoch_stats(cur_epoch)
        self.test_meter.reset()
        return top1_err, top5_err


    def validate_all_subnets(
        self,
        cur_epoch,
        image_size_list=None,
        ks_list=None,
        expand_ratio_list=None,
        depth_list=None,
        width_mult_list=None,
        additional_setting=None,
    ):
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
                                    "d": d,
                                    "e": e,
                                    "ks": k,
                                    "w": w,
                                },
                                # "R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),
                                "D%s-E%s-K%s-W%s" % (d, e, k, w),
                            ]
                        )
        if additional_setting is not None:
            subnet_settings += additional_setting
        
        top1errs, top5errs = [], []
        for setting, name in subnet_settings:
            self.model.set_active_subnet(**setting)
            logger.info("epoch: {} | subnet: {}".format(cur_epoch+1, name))
            top1_err, top5_err = self.test_epoch(cur_epoch)
            top1errs.append(top1_err)
            top5errs.append(top5_err)
        
        self.writer.add_scalar('val/top1_error', list_mean(top1errs), cur_epoch)
        self.writer.add_scalar('val/top5_error', list_mean(top5errs), cur_epoch)
            
        # Saving best model
        if self.best_err > top1_err:
            self.best_err = top1_err
            self.saving(cur_epoch, best=True)


if __name__ == '__main__':
    main()
