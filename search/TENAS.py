import os, time
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


import xnas.core.config as config
import xnas.core.logging as logging
from xnas.core.config import cfg
from xnas.core.trainer import setup_env
from xnas.datasets.loader import getDataLoader
from xnas.search_space.TENAS import get_cell_based_tiny_net, get_search_spaces
from xnas.search_algorithm.TENAS import Linear_Region_Collector, get_ntk_n
from xnas.search_algorithm.TENAS.utils import get_model_infos, round_to


from nas_201_api import NASBench201API as API

# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()

logger = logging.get_logger(__name__)


INF = 1000  # used to mark prunned operators


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model


def prune_func_rank(arch_parameters, model_config, model_config_thin, loader, lrc_model, search_space, precision=10, prune_number=1):
    # arch_parameters now has three dim: cell_type, edge, op
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    init_model(network_origin, cfg.TENAS.INIT)
    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    init_model(network_thin_origin, cfg.TENAS.INIT)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)
    network_thin_origin.set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    prune_number = min(prune_number, alpha_active[0][0].sum()-1)  # adjust prune_number based on current remaining ops on each edge
    ntk_all = []  # (ntk, (edge_idx, op_idx))
    regions_all = []  # (regions, (edge_idx, op_idx))
    choice2regions = {}  # (edge_idx, op_idx): regions
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_edge in range(len(arch_parameters[idx_ct])):
            # edge
            if alpha_active[idx_ct][idx_edge].sum() == 1:
                # only one op remaining
                continue
            for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                # op
                if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                    # this edge-op not pruned yet
                    _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                    _arch_param[idx_ct][idx_edge, idx_op] = -INF
                    # ##### get ntk (score) ########
                    network = get_cell_based_tiny_net(model_config).cuda().train()
                    network.set_alphas(_arch_param)
                    ntk_delta = []
                    repeat = cfg.TENAS.REPEAT
                    for _ in range(repeat):
                        # random reinit
                        init_model(network_origin, cfg.TENAS.INIT+"_fanout" if cfg.TENAS.INIT.startswith('kaiming') else cfg.TENAS.INIT)  # for backward
                        # make sure network_origin and network are identical
                        for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                            param.data.copy_(param_ori.data)
                        network.set_alphas(_arch_param)
                        # NTK cond TODO #########
                        ntk_origin, ntk = get_ntk_n(loader, [network_origin, network], recalbn=0, train_mode=True, num_batch=1)
                        # ####################
                        ntk_delta.append(round((ntk_origin - ntk) / ntk_origin, precision))  # higher the more likely to be prunned
                    ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])  # change of ntk
                    network.zero_grad()
                    network_origin.zero_grad()
                    #############################
                    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                    network_thin_origin.set_alphas(arch_parameters)
                    network_thin_origin.train()
                    network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                    network_thin.set_alphas(_arch_param)
                    network_thin.train()
                    with torch.no_grad():
                        _linear_regions = []
                        repeat = cfg.TENAS.REPEAT
                        for _ in range(repeat):
                            # random reinit
                            init_model(network_thin_origin, cfg.TENAS.INIT+"_fanin" if cfg.TENAS.INIT.startswith('kaiming') else cfg.TENAS.INIT)  # for forward
                            # make sure network_thin and network_thin_origin are identical
                            for param_ori, param in zip(network_thin_origin.parameters(), network_thin.parameters()):
                                param.data.copy_(param_ori.data)
                            network_thin.set_alphas(_arch_param)
                            #####
                            lrc_model.reinit(models=[network_thin_origin, network_thin], seed=cfg.RNG_SEED)
                            _lr, _lr_2 = lrc_model.forward_batch_sample()
                            _linear_regions.append(round((_lr - _lr_2) / _lr, precision))  # change of #Regions, lower the more likely to be prunned
                            lrc_model.clear()
                        linear_regions = np.mean(_linear_regions)
                        regions_all.append([linear_regions, (idx_ct, idx_edge, idx_op)])
                        choice2regions[(idx_ct, idx_edge, idx_op)] = linear_regions
                    #############################
                    torch.cuda.empty_cache()
                    del network_thin
                    del network_thin_origin
                    pbar.update(1)
    ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
    # print("NTK conds:", ntk_all)
    rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]
    for idx, data in enumerate(ntk_all):
        if idx == 0:
            rankings[data[1]] = [idx]
        else:
            if data[0] == ntk_all[idx-1][0]:
                # same ntk as previous
                rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] ]
            else:
                rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] + 1 ]
    regions_all = sorted(regions_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
    # print("#Regions:", regions_all)
    for idx, data in enumerate(regions_all):
        if idx == 0:
            rankings[data[1]].append(idx)
        else:
            if data[0] == regions_all[idx-1][0]:
                # same #Regions as previous
                rankings[data[1]].append(rankings[regions_all[idx-1][1]][1])
            else:
                rankings[data[1]].append(rankings[regions_all[idx-1][1]][1]+1)
    rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    # ascending by sum of two rankings
    rankings_sum = sorted(rankings_list, key=lambda tup: sum(tup[1]), reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    edge2choice = {}  # (cell_idx, edge_idx): list of (cell_idx, edge_idx, op_idx) of length prune_number
    for (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank] in rankings_sum:
        if (cell_idx, edge_idx) not in edge2choice:
            edge2choice[(cell_idx, edge_idx)] = [(cell_idx, edge_idx, op_idx)]
        elif len(edge2choice[(cell_idx, edge_idx)]) < prune_number:
            edge2choice[(cell_idx, edge_idx)].append((cell_idx, edge_idx, op_idx))
    choices_edges = list(edge2choice.values())
    # print("Final Ranking:", rankings_sum)
    # print("Pruning Choices:", choices_edges)
    for choices in choices_edges:
        for (cell_idx, edge_idx, op_idx) in choices:
            arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF

    return arch_parameters, choices_edges


def prune_func_rank_group(arch_parameters, model_config, model_config_thin, loader, lrc_model, search_space, edge_groups=[(0, 2), (2, 5), (5, 9), (9, 14)], num_per_group=2, precision=10):
    # arch_parameters now has three dim: cell_type, edge, op
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    init_model(network_origin, cfg.TENAS.INIT)
    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    init_model(network_thin_origin, cfg.TENAS.INIT)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)
    network_thin_origin.set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    ntk_all = []  # (ntk, (edge_idx, op_idx))
    regions_all = []  # (regions, (edge_idx, op_idx))
    choice2regions = {}  # (edge_idx, op_idx): regions
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    assert edge_groups[-1][1] == len(arch_parameters[0])
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_group in range(len(edge_groups)):
            edge_group = edge_groups[idx_group]
            # print("Pruning cell %s group %s.........."%("normal" if idx_ct == 0 else "reduction", str(edge_group)))
            if edge_group[1] - edge_group[0] <= num_per_group:
                # this group already meets the num_per_group requirement
                pbar.update(1)
                continue
            for idx_edge in range(edge_group[0], edge_group[1]):
                # edge
                for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                    # op
                    if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                        # this edge-op not pruned yet
                        _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                        _arch_param[idx_ct][idx_edge, idx_op] = -INF
                        # ##### get ntk (score) ########
                        network = get_cell_based_tiny_net(model_config).cuda().train()
                        network.set_alphas(_arch_param)
                        ntk_delta = []
                        repeat = cfg.TENAS.REPEAT
                        for _ in range(repeat):
                            # random reinit
                            init_model(network_origin, cfg.TENAS.INIT+"_fanout" if cfg.TENAS.INIT.startswith('kaiming') else cfg.TENAS.INIT)  # for backward
                            # make sure network_origin and network are identical
                            for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                                param.data.copy_(param_ori.data)
                            network.set_alphas(_arch_param)
                            # NTK cond TODO #########
                            ntk_origin, ntk = get_ntk_n(loader, [network_origin, network], recalbn=0, train_mode=True, num_batch=1)
                            # ####################
                            ntk_delta.append(round((ntk_origin - ntk) / ntk_origin, precision))
                        ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])  # change of ntk
                        network.zero_grad()
                        network_origin.zero_grad()
                        #############################
                        network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                        network_thin_origin.set_alphas(arch_parameters)
                        network_thin_origin.train()
                        network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                        network_thin.set_alphas(_arch_param)
                        network_thin.train()
                        with torch.no_grad():
                            _linear_regions = []
                            repeat = cfg.TENAS.REPEAT
                            for _ in range(repeat):
                                # random reinit
                                init_model(network_thin_origin, cfg.TENAS.INIT+"_fanin" if cfg.TENAS.INIT.startswith('kaiming') else cfg.TENAS.INIT)  # for forward
                                # make sure network_thin and network_thin_origin are identical
                                for param_ori, param in zip(network_thin_origin.parameters(), network_thin.parameters()):
                                    param.data.copy_(param_ori.data)
                                network_thin.set_alphas(_arch_param)
                                #####
                                lrc_model.reinit(models=[network_thin_origin, network_thin], seed=cfg.RNG_SEED)
                                _lr, _lr_2 = lrc_model.forward_batch_sample()
                                _linear_regions.append(round((_lr - _lr_2) / _lr, precision))  # change of #Regions
                                lrc_model.clear()
                            linear_regions = np.mean(_linear_regions)
                            regions_all.append([linear_regions, (idx_ct, idx_edge, idx_op)])
                            choice2regions[(idx_ct, idx_edge, idx_op)] = linear_regions
                        #############################
                        torch.cuda.empty_cache()
                        del network_thin
                        del network_thin_origin
                        pbar.update(1)
            # stop and prune this edge group
            ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
            # print("NTK conds:", ntk_all)
            rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]
            for idx, data in enumerate(ntk_all):
                if idx == 0:
                    rankings[data[1]] = [idx]
                else:
                    if data[0] == ntk_all[idx-1][0]:
                        # same ntk as previous
                        rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] ]
                    else:
                        rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] + 1 ]
            regions_all = sorted(regions_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
            # print("#Regions:", regions_all)
            for idx, data in enumerate(regions_all):
                if idx == 0:
                    rankings[data[1]].append(idx)
                else:
                    if data[0] == regions_all[idx-1][0]:
                        # same #Regions as previous
                        rankings[data[1]].append(rankings[regions_all[idx-1][1]][1])
                    else:
                        rankings[data[1]].append(rankings[regions_all[idx-1][1]][1]+1)
            rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
            # ascending by sum of two rankings
            rankings_sum = sorted(rankings_list, key=lambda tup: sum(tup[1]), reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
            choices = [item[0] for item in rankings_sum[:-num_per_group]]
            # print("Final Ranking:", rankings_sum)
            # print("Pruning Choices:", choices)
            for (cell_idx, edge_idx, op_idx) in choices:
                arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF
            # reinit
            ntk_all = []  # (ntk, (edge_idx, op_idx))
            regions_all = []  # (regions, (edge_idx, op_idx))
            choice2regions = {}  # (edge_idx, op_idx): regions

    return arch_parameters


def is_single_path(network):
    arch_parameters = network.get_alphas()
    edge_active = torch.cat([(nn.functional.softmax(alpha, 1) > 0.01).float().sum(1) for alpha in arch_parameters], dim=0)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    return True


def main():
    # PID = os.getpid()
    # assert torch.cuda.is_available(), 'CUDA is not available.'
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # prepare_seed(cfg.RNG_SEED)
    setup_env()


    if cfg.SEARCH.DATASET == 'cifar10' or cfg.SEARCH.DATASET == 'cifar100':
        xshape = (1, 3, 32, 32)
        # train_data, valid_data = getData(cfg.SEARCH.DATASET, cfg.SEARCH.DATAPATH, 0)
        train_loader, valid_loader = getDataLoader(cfg.SEARCH.DATASET, cfg.SEARCH.DATAPATH, cfg.SEARCH.BATCH_SIZE, 0, cfg.DATA_LOADER.NUM_WORKERS)
    elif cfg.SEARCH.DATASET == 'imagenet16':
        xshape = (1, 3, 16, 16)
        train_loader, valid_loader = getDataLoader(cfg.SEARCH.DATASET, cfg.SEARCH.DATAPATH, cfg.SEARCH.BATCH_SIZE, 0, cfg.DATA_LOADER.NUM_WORKERS, use_classes=120)
    elif cfg.SEARCH.DATASET == 'imagenet':
        # NOTE: TENAS resizes imagenet to 32*32
        xshape = (1, 3, 32, 32)
        train_loader, valid_loader = getDataLoader(cfg.SEARCH.DATASET, cfg.SEARCH.DATAPATH, cfg.SEARCH.BATCH_SIZE, 0, cfg.DATA_LOADER.NUM_WORKERS)
        # TODO: fixed the resized ImageNet data.
        print('ImageNet is resized to (32, 32) by the author. This issue will be fixed in the next version.')
        raise NotImplementedError


    search_space = get_search_spaces('cell', cfg.SPACE.NAME)
    if cfg.SPACE.NAME == 'nasbench201':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                              'max_nodes': cfg.SPACE.NODES, 'num_classes': cfg.SEARCH.NUM_CLASSES,
                              'space': search_space,
                              'affine': True, 'track_running_stats': cfg.TENAS.TRACK,
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                   'max_nodes': cfg.SPACE.NODES, 'num_classes': cfg.SEARCH.NUM_CLASSES,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': cfg.TENAS.TRACK,
                                  })
    elif cfg.SPACE.NAME == 'darts':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                              'num_classes': cfg.SEARCH.NUM_CLASSES,
                              'space': search_space,
                              'affine': True, 'track_running_stats': cfg.TENAS.TRACK,
                              'super_type': cfg.TENAS.SUPER_TYPE,
                              'steps': 4,
                              'multiplier': 4,
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 2, 'use_stem': False, 'stem_multiplier': 1,
                                   'max_nodes': cfg.SPACE.NODES, 'num_classes': cfg.SEARCH.NUM_CLASSES,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': cfg.TENAS.TRACK,
                                   'super_type': cfg.TENAS.SUPER_TYPE,
                                   'steps': 4,
                                   'multiplier': 4,
                                  })
    network = get_cell_based_tiny_net(model_config)
    logger.info('model-config : {:}'.format(model_config))
    arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, :] = 0

    # TODO Linear_Region_Collector
    lrc_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3, dataset=cfg.SEARCH.DATASET, data_path=cfg.SEARCH.DATAPATH, seed=cfg.RNG_SEED)

    # ### all params trainable (except train_bn) #########################
    flop, param = get_model_infos(network, xshape)
    logger.info('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.info('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
    if cfg.SPACE.NAME == "nasbench201":
        api = API("./data/NAS-Bench-201-v1_0-e61699.pth")
    else:
        api = None
    # logger.info('{:} create API = {:} done'.format(time_string(), api))

    network = network.cuda()

    genotypes = {}; genotypes['arch'] = {-1: network.genotype()}

    arch_parameters_history = []
    arch_parameters_history_npy = []
    start_time = time.time()
    epoch = -1

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
    arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
    np.save(os.path.join(cfg.OUT_DIR, "arch_parameters_history.npy"), arch_parameters_history_npy)
    while not is_single_path(network):
        epoch += 1
        torch.cuda.empty_cache()

        arch_parameters, op_pruned = prune_func_rank(arch_parameters, model_config, model_config_thin, train_loader, lrc_model, search_space,
                                                     precision=cfg.TENAS.PRECISION,
                                                     prune_number=cfg.TENAS.PRUNE_NUMBER
                                                     )
        # rebuild supernet
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)

        arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        np.save(os.path.join(cfg.OUT_DIR, "arch_parameters_history.npy"), arch_parameters_history_npy)
        genotypes['arch'][epoch] = network.genotype()

        logger.info('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    if cfg.SPACE.NAME == 'darts':
        print("===>>> Prune Edge Groups...")
        arch_parameters = prune_func_rank_group(arch_parameters, model_config, model_config_thin, train_loader, lrc_model, search_space,
                                                edge_groups=[(0, 2), (2, 5), (5, 9), (9, 14)], num_per_group=2,
                                                precision=cfg.TENAS.PRECISION,
                                                )
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)
        arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        np.save(os.path.join(cfg.OUT_DIR, "arch_parameters_history.npy"), arch_parameters_history_npy)

    logger.info('<<<--->>> End: {:}'.format(network.genotype()))
    logger.info('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    end_time = time.time()
    logger.info('\n' + '-'*100)
    logger.info("Time spent: %d s"%(end_time - start_time))
    # check the performance from the architecture dataset
    if api is not None:
        logger.info('{:}'.format(api.query_by_arch(genotypes['arch'][epoch])))



if __name__ == '__main__':   
    main()
