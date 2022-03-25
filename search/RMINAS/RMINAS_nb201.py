import time
import random
import numpy as np

import xnas.search_algorithm.RMINAS.utils.RMI_torch as RMI
from xnas.search_algorithm.RMINAS.sampler.RF_sampling import RF_suggest
import xnas.search_algorithm.RMINAS.sampler.sampling as sampling
    
from xnas.search_space.RMINAS.NB201.utils import *
from nas_201_api import NASBench201API as api

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import xnas.core.config as config
import xnas.core.logging as logging
from xnas.core.config import cfg
from xnas.core.trainer import setup_env


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
    assert cfg.SEARCH.DATASET in ['cifar10', 'cifar100', 'imagenet16_120'], 'dataset error'

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
        model_res = resnet101()
        model_res.load_state_dict(torch.load('xnas/search_algorithm/RMINAS/teacher_model/resnet101_cifar100/resnet101.pth'))
        model_res.cuda()
        
        """selecting well-performed data."""
        with torch.no_grad():
            ce_loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
            more_logits = model_res(more_data_X)
            _, indices = torch.topk(-ce_loss(more_logits, more_data_y).cpu().detach(), cfg.TRAIN.BATCH_SIZE)
        data_y = torch.Tensor([more_data_y[i] for i in indices]).long().cuda()
        data_X = torch.Tensor([more_data_X[i].cpu().numpy() for i in indices]).cuda()
        with torch.no_grad():
            feature_res = model_res.feature_extractor(data_X)
    
    elif cfg.SEARCH.DATASET == 'imagenet16_120':
        import xnas.search_algorithm.RMINAS.utils.imagenet16120_loader as imagenetloader
        from xnas.search_space.RMINAS.NB201.geno import Structure as cellstructure
        from nas_201_api import ResultsCount

        """Data preparing"""
        train_loader, _ = imagenetloader.get_loader(batch_size=cfg.TRAIN.BATCH_SIZE*16)
        target_i = random.randint(0, len(train_loader)-1)
        more_data_X, more_data_y = None, None
        for i, (more_data_X, more_data_y) in enumerate(train_loader):
            if i == target_i:
                break
        more_data_X = more_data_X.cuda()
        more_data_y = more_data_y.cuda()

        """Teacher Network: using best arch searched from cifar10 and weight from nb201."""
        filename = 'xnas/search_algorithm/RMINAS/teacher_model/nb201model_imagenet16120/009930-FULL.pth'
        xdata = torch.load(filename)
        odata  = xdata['full']['all_results'][('ImageNet16-120', 777)]
        result = ResultsCount.create_from_state_dict(odata)
        result.get_net_param()
        arch_config = result.get_config(cellstructure.str2structure) # create the network with params
        net_config = dict2config(arch_config, None)
        network = get_cell_based_tiny_net(net_config)
        network.load_state_dict(result.get_net_param())
        network.cuda()
        
        """selecting well-performed data."""
        with torch.no_grad():
            ce_loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
            _, more_logits = network(more_data_X)
            _, indices = torch.topk(-ce_loss(more_logits, more_data_y).cpu().detach(), cfg.TRAIN.BATCH_SIZE)
        data_y = torch.Tensor([more_data_y[i] for i in indices]).long().cuda()
        data_X = torch.Tensor([more_data_X[i].cpu().numpy() for i in indices]).cuda()
        with torch.no_grad():
            feature_res, _ = network(data_X)
        
    """Codes: build from config file."""
    nb201_api = api('./data/NAS-Bench-201-v1_0-e61699.pth')
    
    RFS = RF_suggest(space='nasbench201', logger=logger, api=nb201_api, thres_rate=cfg.RMINAS.RF_THRESRATE, seed=cfg.RNG_SEED)

    # loss function
    loss_fun_cka = CKA_loss(data_X.size()[0])
    loss_fun_cka = loss_fun_cka.requires_grad_()
    loss_fun_cka.cuda()
    loss_fun_log = torch.nn.CrossEntropyLoss().cuda()
        
    def train_arch(arch_index):                
        # get arch
        arch_config = {
            'name': 'infer.tiny', 
            'C': 16, 'N': 5, 
            'arch_str':nb201_api.arch(arch_index), 
            'num_classes': cfg.SEARCH.NUM_CLASSES}
        net_config = dict2config(arch_config, None)
        model = get_cell_based_tiny_net(net_config)
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
            
            features, logits = model(data_X)
            loss_logits = loss_fun_log(logits, data_y)
            loss_cka = loss_fun_cka(features, feature_res)
            loss = cfg.RMINAS.LOSS_BETA * loss_cka + (1-cfg.RMINAS.LOSS_BETA)*loss_logits
            loss.backward()

            optimizer.step()
            
            if cur_epoch == cfg.OPTIM.MAX_EPOCH:
                logger.info('Arch:{} Loss:{}'.format(str(arch_index), str(loss.cpu().detach().numpy())))
                return loss.cpu().detach().numpy()
        
    start_time = time.time()
    trained_loss = []
    
    # ====== Warmup ======
    warmup_samples = RFS.warmup_samples(cfg.RMINAS.RF_WARMUP)
    logger.info("Warming up with {} archs".format(cfg.RMINAS.RF_WARMUP))
    for arch_index in warmup_samples:
        mixed_loss = train_arch(arch_index)
        mixed_loss = np.inf if np.isnan(mixed_loss) else mixed_loss
        trained_loss.append(mixed_loss)
        arch_arr = sampling.genostr2array(nb201_api.arch(arch_index))
        RFS.trained_arch.append({'arch':arch_arr, 'loss':mixed_loss})
        RFS.trained_arch_index.append(arch_index)
#         print(arch_index, mixed_loss)
    RFS.Warmup()
    logger.info('warmup time cost: {}'.format(str(time.time() - start_time)))
    
    # ====== RF Sampling ======
    sampling_time = time.time()
    sampling_cnt= 0
    while sampling_cnt < cfg.RMINAS.RF_SUCC:
        arch_index = RFS.fitting_samples()
        assert arch_index not in list(RFS.trained_arch_index), "RFS.trained_arch_index error"
        mixed_loss = train_arch(arch_index)
        mixed_loss = np.inf if np.isnan(mixed_loss) else mixed_loss
        RFS.trained_arch_index.append(arch_index)
        trained_loss.append(mixed_loss)
        arch_arr = sampling.genostr2array(nb201_api.arch(arch_index))
        RFS.trained_arch.append({'arch':arch_arr, 'loss':mixed_loss})
#         print(arch_index, mixed_loss)
        sampling_cnt += RFS.Fitting()
    if sampling_cnt >= cfg.RMINAS.RF_SUCC:
        logger.info('successfully sampling good archs for {} times'.format(sampling_cnt))
    else:
        logger.info('failed sampling good archs for only {} times'.format(sampling_cnt))
    logger.info('RF sampling time cost:{}'.format(str(time.time() - sampling_time)))
    
    # ====== Evaluation ======
    logger.info('Total time cost: {}'.format(str(time.time() - start_time)))
    logger.info('Actual training times: {}'.format(len(RFS.trained_arch_index)))
    logger.info('Searched architecture:\n{}'.format(str(RFS.optimal_arch(method='sum', top=50))))
    # logger.info('Searched architecture:\n{}'.format(str(RFS.optimal_arch(method='greedy', top=50))))

if __name__ == '__main__':
    main()
