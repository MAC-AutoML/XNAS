"""AttentiveNAS supernet training"""

import os
import random
import operator

import torch
import torch.nn as nn

import xnas.core.config as config
from xnas.datasets.loader import get_normal_dataloader
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# AttentiveNAS
from xnas.runner.trainer import Trainer
from xnas.runner.scheduler import adjust_learning_rate_per_batch
from xnas.algorithms.AttentiveNAS.sampler import ArchSampler
from xnas.spaces.OFA.utils import list_mean
from xnas.spaces.BigNAS.utils import init_model


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main(local_rank, world_size):
    setup_env()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    # Network
    net = space_builder().to(local_rank)
    init_model(net)
    # Loss function
    criterion = criterion_builder()
    soft_criterion = criterion_builder('kl_soft')
    
    # Data loaders
    [train_loader, valid_loader] = get_normal_dataloader()
    
    # Optimizers
    net_params = [
        # parameters with weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="exclude"), "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        # parameters without weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="include") , "weight_decay": 0}, 
    ]
    optimizer = optimizer_builder("SGD", net_params)
    
    # sampler for AttentiveNAS
    sampler = ArchSampler(
        cfg.ATTENTIVENAS.SAMPLER.MAP_PATH, cfg.ATTENTIVENAS.SAMPLER.DISCRETIZE_STEP, net, None
    )
    
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)
    
    # Initialize Recorder
    attnas_trainer = AttentivenasTrainer(
        model=net,
        criterion=criterion,
        soft_criterion=soft_criterion,
        sampler=sampler,
        optimizer=optimizer,
        lr_scheduler=None,
        train_loader=train_loader,
        test_loader=valid_loader,
    )
    
    # Resume
    start_epoch = attnas_trainer.loading() if cfg.SEARCH.AUTO_RESUME else 0
    
    # Training
    logger.info("Start AttentiveNAS training.")
    dist.barrier()
    attnas_trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.WARMUP_EPOCH+cfg.OPTIM.MAX_EPOCH):
        attnas_trainer.train_epoch(cur_epoch, rank=local_rank)
        if local_rank == 0:
            if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
                attnas_trainer.validate(cur_epoch, local_rank)
    attnas_trainer.finish()
    dist.barrier()
    torch.cuda.empty_cache()


class AttentivenasTrainer(Trainer):
    """Trainer for AttentiveNAS."""
    def __init__(self, model, criterion, soft_criterion, sampler, optimizer, lr_scheduler, train_loader, test_loader):
        super().__init__(model, criterion, optimizer, lr_scheduler, train_loader, test_loader)
        self.sandwich_sample_num = max(2, cfg.ATTENTIVENAS.SANDWICH_NUM)    # containing max & min
        self.soft_criterion = soft_criterion
        self.sampler = sampler

    def train_epoch(self, cur_epoch, rank=0):
        self.model.train()
        # lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        # self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        self.train_loader.sampler.set_epoch(cur_epoch)  # DDP
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            
            # Adjust lr per iter
            cur_lr = adjust_learning_rate_per_batch(
                epoch=cur_epoch,
                n_iter=len(self.train_loader),
                iter=cur_iter,
                warmup=(cur_epoch < cfg.OPTIM.WARMUP_EPOCH),
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = cur_lr
            # self.writer.add_scalar('train/lr', cur_lr, cur_step)
            
            self.optimizer.zero_grad()
            
            ## Sandwich Rule ##
            # Step 1. Largest network sampling & regularization
            self.model.module.sample_max_subnet()
            self.model.module.set_dropout_rate(cfg.TRAIN.DROP_PATH_PROB, cfg.ATTENTIVENAS.DROP_CONNECT)
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            
            with torch.no_grad():
                soft_logits = preds.clone().detach()
            
            # Step 2. sample smaller networks
            self.model.module.set_dropout_rate(0, 0)
            for arch_id in range(1, self.sandwich_sample_num):
                if arch_id == self.sandwich_sample_num - 1:
                    self.model.module.sample_min_subnet()
                else:
                    if self.sampler is not None:
                        sampling_method = cfg.ATTENTIVENAS.SAMPLER.METHOD
                        if sampling_method in ['bestup', 'worstup']:
                            target_flops = self.sampler.sample_one_target_flops()
                            candidate_archs = self.sampler.sample_archs_according_to_flops(
                                target_flops, n_samples=cfg.ATTENTIVENAS.SAMPLER.NUM_TRIALS
                            )
                            my_pred_accs = []
                            for arch in candidate_archs:
                                self.model.module.set_active_subnet(**arch)
                                with torch.no_grad():
                                    my_pred_accs.append(-1.0 * self.criterion(self.model(inputs), labels))
                            if sampling_method == 'bestup':
                                idx, _ = max(enumerate(my_pred_accs), key=operator.itemgetter(1))                          
                            else:
                                idx, _ = min(enumerate(my_pred_accs), key=operator.itemgetter(1))                          
                            self.model.module.set_active_subnet(**candidate_archs[idx])  #reset
                    else:
                        subnet_seed = int("%d%.3d%.3d" % (cur_step, arch_id, 0))
                        random.seed(subnet_seed)
                        self.model.module.sample_active_subnet()
                
                preds = self.model(inputs)
                
                if self.soft_criterion is not None:
                    loss = self.soft_criterion(preds, soft_logits)
                else:
                    loss = self.criterion(preds, labels)
                loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()
            
            # calculating errors. The source code of AttentiveNAS uses statistics of the smallest network and XNAS follows.
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            self.train_meter.iter_toc()
            self.train_meter.update_stats(top1_err, top5_err, loss, cur_lr, inputs.size(0) * cfg.NUM_GPUS)
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            # self.writer.add_scalar('train/loss', i_loss, cur_step)
            # self.writer.add_scalar('train/top1_error', i_top1err, cur_step)
            # self.writer.add_scalar('train/top5_error', i_top5err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        # self.lr_scheduler.step()
        # Saving checkpoint
        if rank==0 and (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)
    
    @torch.no_grad()
    def test_epoch(self, subnet, cur_epoch, rank=0):
        subnet.eval()        
        self.test_meter.reset(True)
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            preds = subnet(inputs)
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


    def validate(self, cur_epoch, rank, bn_calibration=True):
        subnets_to_be_evaluated = {
            'attentive_nas_min_net': {},
            'attentive_nas_max_net': {},
        }
        
        top1_list, top5_list = [], []
        with torch.no_grad():
            for net_id in subnets_to_be_evaluated:
                if net_id == 'attentive_nas_min_net': 
                    self.model.module.sample_min_subnet()
                elif net_id == 'attentive_nas_max_net':
                    self.model.module.sample_max_subnet()
                elif net_id.startswith('attentive_nas_random_net'):
                    self.model.module.sample_active_subnet()
                else:
                    self.model.module.set_active_subnet(
                        subnets_to_be_evaluated[net_id]['resolution'],
                        subnets_to_be_evaluated[net_id]['width'],
                        subnets_to_be_evaluated[net_id]['depth'],
                        subnets_to_be_evaluated[net_id]['kernel_size'],
                        subnets_to_be_evaluated[net_id]['expand_ratio'],
                    )

                subnet = self.model.module.get_active_subnet()
                subnet.to(rank)
                logger.info("evaluating subnet {}".format(net_id))
                
                if bn_calibration:
                    subnet.eval()
                    logger.info("Calibrating BN running statistics.")
                    subnet.reset_running_stats_for_calibration()
                    for cur_iter, (inputs, _) in enumerate(self.train_loader):
                        if cur_iter >= cfg.ATTENTIVENAS.POST_BN_CALIBRATION_BATCH_NUM:
                            break
                        inputs = inputs.to(rank)
                        subnet(inputs)      # forward only                
                
                top1_err, top5_err = self.test_epoch(subnet, cur_epoch, rank)
                top1_list.append(top1_err), top5_list.append(top5_err)
            logger.info("Average@all_subnets top1_err:{} top5_err:{}".format(list_mean(top1_list), list_mean(top5_list)))
            if self.best_err > list_mean(top1_list):
                self.best_err = list_mean(top1_list)
                self.saving(cur_epoch, best=True)


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23333'
    
    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()
    
    mp.spawn(main, nprocs=cfg.NUM_GPUS, args=(cfg.NUM_GPUS,), join=True)
