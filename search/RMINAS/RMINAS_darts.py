import time
import numpy as np
import time

import xnas.search_algorithm.RMINAS.utils.RMI_torch as RMI
from xnas.search_algorithm.RMINAS.sampler.RF_sampling import RF_suggest
import xnas.search_algorithm.RMINAS.sampler.sampling_darts as sampling

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import xnas.core.config as config
import xnas.core.logging as logging
from xnas.core.config import cfg
from xnas.core.trainer import setup_env

from xnas.search_space.RMINAS.DARTS.darts_cnn import AugmentCNN, geno_from_alpha, reformat_DARTS


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
    logger = logging.get_logger(__name__)
    
    # Load config and check
    config.load_cfg_fom_args()
    config.assert_and_infer_cfg()
    cfg.freeze()
    
    setup_env()
    
    print(cfg.SEARCH.DATASET)
    # assert cfg.SEARCH.DATASET in ['cifar10', 'cifar100'], 'dataset error'
    assert cfg.SEARCH.DATASET in ['cifar10', 'cifar100', 'imagenet'], 'dataset error'
    if cfg.SEARCH.DATASET == 'imagenet':
        print('='*30+' NOTE '+'='*30)
        print('Our method does not directly search in ImageNet.')
        print('Only partial tests have been conducted, please use with caution.')
        print('='*66)

    if cfg.SEARCH.DATASET == 'cifar10':
        from xnas.search_algorithm.RMINAS.utils.loader import cifar10_data
        import xnas.search_algorithm.RMINAS.teacher_model.resnet20_cifar10.resnet as resnet
        """Data preparing"""
        more_data_X, more_data_y = cifar10_data(cfg.TRAIN.BATCH_SIZE, cfg.DATA_LOADER.NUM_WORKERS)

        """ResNet codes"""
        checkpoint_res = torch.load('xnas/search_algorithm/RMINAS/teacher_model/resnet20_cifar10/resnet20.th')
        model_res = torch.nn.DataParallel(resnet.__dict__['resnet20']())
        model_res.cuda()
        model_res.load_state_dict(checkpoint_res['state_dict'])
        
        """selecting well-performed data."""
        with torch.no_grad():
            ce_loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
            more_logits = model_res(more_data_X)
            _, indices = torch.topk(-ce_loss(more_logits, more_data_y).cpu().detach(), cfg.TRAIN.BATCH_SIZE)
        data_y = torch.Tensor([more_data_y[i] for i in indices]).long().cuda()
        data_X = torch.Tensor([more_data_X[i].cpu().numpy() for i in indices]).cuda()
        with torch.no_grad():
            feature_res = model_res.module.feature_extractor(data_X)

    elif cfg.SEARCH.DATASET == 'cifar100':
        from xnas.search_algorithm.RMINAS.utils.loader import cifar100_data
        from xnas.search_algorithm.RMINAS.teacher_model.resnet101_cifar100.resnet import resnet101
        """Data preparing"""
        more_data_X, more_data_y = cifar100_data(cfg.TRAIN.BATCH_SIZE, cfg.DATA_LOADER.NUM_WORKERS)
        
        """ResNet codes"""
        net = resnet101()
        net.load_state_dict(torch.load('xnas/search_algorithm/RMINAS/teacher_model/resnet101_cifar100/resnet101.pth'))
        net.cuda()
        
        """selecting well-performed data."""
        with torch.no_grad():
            ce_loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
            more_logits = net(more_data_X)
            _, indices = torch.topk(-ce_loss(more_logits, more_data_y).cpu().detach(), cfg.TRAIN.BATCH_SIZE)
        data_y = torch.Tensor([more_data_y[i] for i in indices]).long().cuda()
        data_X = torch.Tensor([more_data_X[i].cpu().numpy() for i in indices]).cuda()
        with torch.no_grad():
            feature_res = net.feature_extractor(data_X)

    elif cfg.SEARCH.DATASET == 'imagenet':
        from xnas.search_algorithm.RMINAS.utils.loader import imagenet_data
        import xnas.search_algorithm.RMINAS.teacher_model.fbresnet_imagenet.fbresnet as fbresnet
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

    RFS = RF_suggest(space='darts', logger=logger, thres_rate=cfg.RMINAS.RF_THRESRATE, seed=cfg.RNG_SEED)
    
    # loss function
    loss_fun_cka = CKA_loss(data_X.size()[0])
    loss_fun_cka = loss_fun_cka.requires_grad_()
    loss_fun_cka.cuda()
    loss_fun_log = torch.nn.CrossEntropyLoss().cuda()
        
    def train_arch(genotype):
        s_time = time.time()
        model = AugmentCNN(
            cfg.SEARCH.IM_SIZE, 
            cfg.SEARCH.INPUT_CHANNEL, 
            cfg.TRAIN.CHANNELS,
            cfg.SEARCH.NUM_CLASSES, 
            cfg.TRAIN.LAYERS, 
            False, # don't use auxiliary head
            genotype)
        model.cuda()
        model.train()
        
        # weights optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), 
            cfg.OPTIM.BASE_LR, 
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY)

        for cur_epoch in range(1, cfg.OPTIM.MAX_EPOCH+1):
            optimizer.zero_grad()

            features, logits, aux_logits = model(data_X)
            loss_cka = loss_fun_cka(features, feature_res)
            loss_logits = loss_fun_log(logits, data_y)
            loss = cfg.RMINAS.LOSS_BETA * loss_cka + (1-cfg.RMINAS.LOSS_BETA)*loss_logits
            loss.backward()

            optimizer.step()

            if cur_epoch == cfg.OPTIM.MAX_EPOCH:
                logger.info("training arch cost: {}".format(time.time()-s_time))
                return loss.cpu().detach().numpy()
    
    start_time = time.time()
    trained_arch, trained_loss = [], []

    # ====== Warmup ======
    warmup_samples = RFS.warmup_samples(cfg.RMINAS.RF_WARMUP)
    logger.info("Warming up with {} archs".format(cfg.RMINAS.RF_WARMUP))
    for sample in warmup_samples:
        sample_alpha = sampling.ransug2alpha(sample)  # shape=(28, 8)
        sample_geno = geno_from_alpha(sample_alpha)  # type=Genotype
        # if cfg.SEARCH.DATASET == 'imagenet' :
        #     sample_geno = reformat_DARTS(sample_geno)
        mixed_loss = train_arch(sample_geno)
        mixed_loss = np.inf if np.isnan(mixed_loss) else mixed_loss
        trained_arch.append(str(sample_geno))
        trained_loss.append(mixed_loss)
        RFS.trained_arch.append({'arch':sample, 'loss':mixed_loss})
    RFS.Warmup()
    logger.info('warmup time cost: {}'.format(str(time.time() - start_time)))
    
    # ====== RF Sampling ======
    sampling_time = time.time()
    sampling_cnt = 0
    while sampling_cnt < cfg.RMINAS.RF_SUCC:
        sample = RFS.fitting_samples()
        sample_alpha = sampling.ransug2alpha(sample)  # shape=(28, 8)
        sample_geno = geno_from_alpha(sample_alpha)  # type=Genotype
        # if cfg.SEARCH.DATASET == 'imagenet' :
        #     sample_geno = reformat_DARTS(sample_geno)
        mixed_loss = train_arch(sample_geno)
        mixed_loss = np.inf if np.isnan(mixed_loss) else mixed_loss
        trained_arch.append(str(sample_geno))
        trained_loss.append(mixed_loss)
        RFS.trained_arch.append({'arch':sample, 'loss':mixed_loss})
        sampling_cnt += RFS.Fitting()
    if sampling_cnt >= cfg.RMINAS.RF_SUCC:
        logger.info('successfully sampling good archs for {} times'.format(sampling_cnt))
    else:
        logger.info('failed sampling good archs for only {} times'.format(sampling_cnt))
    logger.info('RF sampling time cost: {}'.format(str(time.time() - sampling_time)))
    
    # ====== Evaluation ======
    logger.info('Total time cost:{}'.format(str(time.time() - start_time)))
    logger.info('Actual training times: {}'.format(len(trained_arch)))
    op_sample = RFS.optimal_arch(method='sum', top=50)
    op_alpha = torch.from_numpy(np.r_[op_sample, op_sample])
    op_geno = reformat_DARTS(geno_from_alpha(op_alpha))
    logger.info('Searched architecture@top50:\n{}'.format(str(op_geno)))

if __name__ == "__main__":
    main()

