import time
import numpy as np

import torch

import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import setup_env, space_builder, optimizer_builder

import xnas.algorithms.RMINAS.sampler.sampling as sampling
from xnas.algorithms.RMINAS.utils.RMI_torch import RMI_loss
from xnas.algorithms.RMINAS.sampler.RF_sampling import RF_suggest
from xnas.algorithms.RMINAS.utils.random_data import get_random_data

from xnas.spaces.DARTS.utils import geno_from_alpha, reformat_DARTS
from xnas.spaces.NASBench201.utils import dict2config, get_cell_based_tiny_net, CellStructure


config.load_configs()
logger = logging.get_logger(__name__)

# RMINAS hyperparameters initialization
nb201_api = None
RF_space = None

def rminas_hp_builder():
    global nb201_api, RF_space
    if cfg.SPACE.NAME == 'infer_nb201':
        from nas_201_api import NASBench201API as api
        nb201_api = api(cfg.BENCHMARK.NB201PATH)
        RF_space = 'nasbench201'
    elif cfg.SPACE.NAME == 'infer_darts':
        RF_space = 'darts'


def main():    
    setup_env()
    
    rminas_hp_builder()
    
    assert cfg.SPACE.NAME in ['infer_nb201', 'infer_darts']
    assert cfg.LOADER.DATASET in ['cifar10', 'cifar100', 'imagenet', 'imagenet16_120'], 'dataset error'

    if cfg.LOADER.DATASET == 'cifar10':
        from xnas.algorithms.RMINAS.teacher_model.resnet20_cifar10.resnet import resnet20
        checkpoint_res = torch.load('xnas/algorithms/RMINAS/teacher_model/resnet20_cifar10/resnet20.th')
        network = torch.nn.DataParallel(resnet20())
        network.load_state_dict(checkpoint_res['state_dict'])
        network = network.module

    elif cfg.LOADER.DATASET == 'cifar100':
        from xnas.algorithms.RMINAS.teacher_model.resnet101_cifar100.resnet import resnet101
        network = resnet101()
        network.load_state_dict(torch.load('xnas/algorithms/RMINAS/teacher_model/resnet101_cifar100/resnet101.pth'))

    elif cfg.LOADER.DATASET == 'imagenet':
        assert cfg.SPACE.NAME == 'infer_darts'
        logger.warning('Our method does not directly search in ImageNet.')
        logger.warning('Only partial tests have been conducted, please use with caution.')
        import xnas.algorithms.RMINAS.teacher_model.fbresnet_imagenet.fbresnet as fbresnet
        network = fbresnet.fbresnet152()
    
    elif cfg.LOADER.DATASET == 'imagenet16_120':
        assert cfg.SPACE.NAME == 'infer_nb201'
        from nas_201_api import ResultsCount

        """Teacher Network: using best arch searched from cifar10 and weight from nb201."""
        filename = 'xnas/algorithms/RMINAS/teacher_model/nb201model_imagenet16120/009930-FULL.pth'
        xdata = torch.load(filename)
        odata  = xdata['full']['all_results'][('ImageNet16-120', 777)]
        result = ResultsCount.create_from_state_dict(odata)
        result.get_net_param()
        arch_config = result.get_config(CellStructure.str2structure) # create the network with params
        net_config = dict2config(arch_config, None)
        network = get_cell_based_tiny_net(net_config) 
        network.load_state_dict(result.get_net_param())
    
    network.cuda()
    
    """selecting well-performed data."""
    more_data_X, more_data_y = get_random_data(cfg.LOADER.BATCH_SIZE, cfg.LOADER.DATASET)
    with torch.no_grad():
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        more_logits = network(more_data_X)
        _, indices = torch.topk(-ce_loss(more_logits, more_data_y).cpu().detach(), cfg.LOADER.BATCH_SIZE)
    data_y = torch.Tensor([more_data_y[i] for i in indices]).long().cuda()
    data_X = torch.Tensor([more_data_X[i].cpu().numpy() for i in indices]).cuda()
    with torch.no_grad():
        feature_res = network.feature_extractor(data_X)
    
    RFS = RF_suggest(space=RF_space, logger=logger, api=nb201_api, thres_rate=cfg.RMINAS.RF_THRESRATE, seed=cfg.RNG_SEED)

    # loss function
    loss_fun_cka = RMI_loss(data_X.size()[0])
    loss_fun_cka = loss_fun_cka.requires_grad_()
    loss_fun_cka.cuda()
    loss_fun_log = torch.nn.CrossEntropyLoss().cuda()
        
    def train_arch(modelinfo):      
        if cfg.SPACE.NAME == 'infer_nb201':
            # get arch
            arch_config = {
                'name': 'infer.tiny', 
                'C': 16, 'N': 5, 
                'arch_str':nb201_api.arch(modelinfo), 
                'num_classes': cfg.LOADER.NUM_CLASSES}
            net_config = dict2config(arch_config, None)
            model = get_cell_based_tiny_net(net_config).cuda()
        elif cfg.SPACE.NAME == 'infer_darts':
            cfg.TRAIN.GENOTYPE = str(modelinfo)
            model = space_builder().cuda()
        
        model.train()
        # weights optimizer
        optimizer = optimizer_builder("SGD", model.parameters())

        for cur_epoch in range(1, cfg.OPTIM.MAX_EPOCH+1):
            optimizer.zero_grad()
            
            features, logits = model.forward_with_features(data_X)
            loss_cka = loss_fun_cka(features, feature_res)
            loss_logits = loss_fun_log(logits, data_y)
            loss = cfg.RMINAS.LOSS_BETA * loss_cka + (1-cfg.RMINAS.LOSS_BETA)*loss_logits
            loss.backward()

            optimizer.step()
            
            if cur_epoch == cfg.OPTIM.MAX_EPOCH:
                return loss.cpu().detach().numpy()
    
    trained_arch_darts, trained_loss = [], []
    def train_procedure(sample):
        if cfg.SPACE.NAME == 'infer_nb201':
            mixed_loss = train_arch(sample)
            mixed_loss = np.inf if np.isnan(mixed_loss) else mixed_loss
            trained_loss.append(mixed_loss)
            arch_arr = sampling.nb201genostr2array(nb201_api.arch(sample))
            RFS.trained_arch.append({'arch':arch_arr, 'loss':mixed_loss})
            RFS.trained_arch_index.append(sample)
        elif cfg.SPACE.NAME == 'infer_darts':
            sample_geno = geno_from_alpha(sampling.darts_sug2alpha(sample))  # type=Genotype
            trained_arch_darts.append(str(sample_geno))
            mixed_loss = train_arch(sample_geno)
            mixed_loss = np.inf if np.isnan(mixed_loss) else mixed_loss
            trained_loss.append(mixed_loss)
            RFS.trained_arch.append({'arch':sample, 'loss':mixed_loss})
        logger.info("sample: {}, loss:{}".format(sample, mixed_loss))
    
    start_time = time.time()
    # ====== Warmup ======
    warmup_samples = RFS.warmup_samples(cfg.RMINAS.RF_WARMUP)
    logger.info("Warming up with {} archs".format(cfg.RMINAS.RF_WARMUP))
    for sample in warmup_samples:
        train_procedure(sample)
    RFS.Warmup()
    logger.info('warmup time cost: {}'.format(str(time.time() - start_time)))
    
    # ====== RF Sampling ======
    sampling_time = time.time()
    sampling_cnt= 0
    while sampling_cnt < cfg.RMINAS.RF_SUCC:
        sample = RFS.fitting_samples()
        train_procedure(sample)
        sampling_cnt += RFS.Fitting()
    if sampling_cnt >= cfg.RMINAS.RF_SUCC:
        logger.info('successfully sampling good archs for {} times'.format(sampling_cnt))
    else:
        logger.info('failed sampling good archs for only {} times'.format(sampling_cnt))
    logger.info('RF sampling time cost:{}'.format(str(time.time() - sampling_time)))
    
    # ====== Evaluation ======
    logger.info('Total time cost: {}'.format(str(time.time() - start_time)))
    logger.info('Actual training times: {}'.format(len(RFS.trained_arch_index)))
    if cfg.SPACE.NAME == 'infer_nb201':
        logger.info('Searched architecture:\n{}'.format(str(RFS.optimal_arch(method='sum', top=50))))
        logger.info('Searched architecture:\n{}'.format(str(RFS.optimal_arch(method='greedy', top=50))))
    elif cfg.SPACE.NAME == 'infer_darts':
        op_sample = RFS.optimal_arch(method='sum', top=50)
        op_alpha = torch.from_numpy(np.r_[op_sample, op_sample])
        op_geno = reformat_DARTS(geno_from_alpha(op_alpha))
        logger.info('Searched architecture@top50:\n{}'.format(str(op_geno)))


if __name__ == '__main__':
    main()
