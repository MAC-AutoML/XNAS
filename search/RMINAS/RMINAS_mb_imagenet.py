import numpy as np
import random
import os
import time

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

import xnas.core.logging as logging
import xnas.core.config as config

from xnas.core.utils import one_hot_to_index
from xnas.core.trainer import setup_env
from xnas.core.config import cfg

from xnas.search_space.RMINAS.MBConv.mb_v3_cnn import MobileNetV3
import xnas.search_algorithm.RMINAS.utils.RMI_torch as RMI
from xnas.search_algorithm.RMINAS.sampler.RF_sampling import RF_suggest

from xnas.search_algorithm.RMINAS.utils.loader import imagenet_data

import xnas.search_algorithm.RMINAS.teacher_model.fbresnet_imagenet.fbresnet as fbresnet


# NOTE: this code is not fully tested.
# OBSERVE_EPO = 250
# RF_WARMUP = 200


class CKA_loss(nn.Module):
    def __init__(self, datasize):
        super(CKA_loss, self).__init__()
        self.datasize = datasize

    def forward(self, features_1, features_2):
        s = []
        for i in range(len(features_1)):
            s.append(RMI.tensor_cka(RMI.tensor_gram_linear(features_1[i].view(self.datasize, -1)), RMI.tensor_gram_linear(features_2[i].view(self.datasize, -1))))
        return torch.sum(3 - s[0] - s[1] - s[2])

def main():    
    # Load config and check
    config.load_cfg_fom_args()
    config.assert_and_infer_cfg()
    cfg.freeze()
    
    setup_env()

    logger = logging.get_logger(__name__)
    
    """Data preparing"""
    more_data_X, more_data_y = imagenet_data(cfg.TRAIN.BATCH_SIZE, cfg.DATA_LOADER.NUM_WORKERS, '/media/DATASET/ILSVRC2012/')
    
    """ResNet codes"""
    model_res = fbresnet.fbresnet152()
    model_res.cuda()
    
    """selecting well-performed data."""
    with torch.no_grad():
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        more_logits = model_res(more_data_X)
        _, indices = torch.topk(-ce_loss(more_logits, more_data_y).cpu().detach(), cfg.TRAIN.BATCH_SIZE)

    data_y = torch.Tensor([more_data_y[i] for i in indices]).long().cuda()
    data_X = torch.Tensor([more_data_X[i].cpu().numpy() for i in indices]).cuda()
    
    with torch.no_grad():
        feature_res = model_res.features_extractor(data_X)
    
    RFS = RF_suggest(space='mb', logger=logger, thres_rate=cfg.RMINAS.RF_THRESRATE, seed=cfg.RNG_SEED)
    
    # loss function
    loss_fun_cka = CKA_loss(data_X.size()[0])
    loss_fun_cka = loss_fun_cka.requires_grad_()
    loss_fun_cka.cuda()
    loss_fun_log = torch.nn.CrossEntropyLoss().cuda()
    
    
    def train_arch(sample):
        
        model = MobileNetV3(n_classes=1000)
        model.cuda()
        
        w_optim = torch.optim.SGD(model.parameters(),
                                  cfg.OPTIM.BASE_LR,
                                  momentum=cfg.OPTIM.MOMENTUM,
                                  weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, cfg.OPTIM.MAX_EPOCH, eta_min=cfg.OPTIM.MIN_LR)
        
        model.train()
        
        logger.info("Sampling: {}".format(one_hot_to_index(sample)))
        for cur_epoch in range(1, cfg.OPTIM.MAX_EPOCH+1):
            
            lr = w_optim.param_groups[0]['lr']

            logits, features = model(data_X, sample)
            loss_cka = loss_fun_cka(features, feature_res)
            loss_logits = loss_fun_log(logits, data_y)
            loss = cfg.RMINAS.LOSS_BETA * loss_cka + (1-cfg.RMINAS.LOSS_BETA)*loss_logits

            w_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
            w_optim.step()

            lr_scheduler.step()
            
            if cur_epoch == cfg.OPTIM.MAX_EPOCH:
                return loss.cpu().detach().numpy()
    
    start_time = time.time()

    # ====== Warmup ======
    warmup_samples = RFS.warmup_samples(cfg.RMINAS.RF_WARMUP)
    logger.info("Warming up with {} archs".format(cfg.RMINAS.RF_WARMUP))
    for sample in warmup_samples:
        mixed_loss = train_arch(sample)
        mixed_loss = np.inf if np.isnan(mixed_loss) else mixed_loss
        RFS.trained_arch.append({'arch':sample, 'loss':mixed_loss})
#         print(str(sample_geno), mixed_loss)
    RFS.Warmup()
    logger.info('warmup time cost: {}'.format(str(time.time() - start_time)))
    
    # ====== RF Sampling ======
    sampling_time = time.time()
    sampling_cnt = 0
    while sampling_cnt < cfg.RMINAS.RF_SUCC:
        sample = RFS.fitting_samples()
        mixed_loss = train_arch(sample)
        mixed_loss = np.inf if np.isnan(mixed_loss) else mixed_loss
        RFS.trained_arch.append({'arch':sample, 'loss':mixed_loss})
#         print(str(sample_geno), mixed_loss)
        sampling_cnt += RFS.Fitting()
    if sampling_cnt >= cfg.RMINAS.RF_SUCC:
        logger.info('successfully sampling good archs for {} times'.format(sampling_cnt))
    else:
        logger.info('failed sampling good archs for only {} times'.format(sampling_cnt))
    logger.info('RF sampling time cost: {}'.format(str(time.time() - sampling_time)))
    
    # ====== Evaluation ======
    logger.info('Total time cost:{}'.format(str(time.time() - start_time)))
    logger.info('Actual training times: {}'.format(len(RFS.trained_arch)))
    op_sample = RFS.optimal_arch(method='sum', top=30)
    logger.info('Searched architecture@top50:\n{}'.format(str(op_sample)))
#     logger.info(model.genotype(torch.Tensor(op_sample)))

if __name__ == "__main__":
    main()
