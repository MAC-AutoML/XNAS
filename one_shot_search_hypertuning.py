""" Search cell """
import os
import hyperopt
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config.config import SearchConfig
import utils.utils as utils
from model.darts_cnn import SelectSearchCNN, NASBench201CNN
from model.mb_v3_cnn import get_super_net
from datasets import get_data
from search_algorithm import Category_MDENAS, Category_DDPNAS, Category_SNG, Category_ASNG, \
    Category_Dynamic_ASNG, Category_Dynamic_SNG, Category_Dynamic_SNG_V3, Category_DDPNAS_V2, \
    Category_DDPNAS_V3
from utils import genotypes
import random
import json
from network_generator import *
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def mkdir(path):
    if os.path.isdir(path):
        return
    else:
        os.mkdir(path)


device = torch.device("cuda")
config_path = os.path.join('/userhome/project/Auto_NAS_V2/experiments/hyper_tunning', 'faster_' + str(time.time()))
# tensorboard
writer = SummaryWriter(logdir=os.path.join(config_path, "tb"))
mkdir(config_path)

logger = utils.get_logger(os.path.join(config_path, "logger.log"))


def main(init_channels=16, layers=5,
         w_lr=0.1, w_momentum=0.9, w_weight_decay=3e-4, w_lr_step=20,
         datset_split=10, warm_up_epochs=0,
         pruning_step=3, gamma=0.8):
    logger.info("init_channels:{},layers:{},"
                "w_lr:{},w_momentum:{},w_weight_decay:{},w_lr_step:{},"
                "datset_split:{},warm_up_epochs:{},"
                "pruning_step:{},gamma:{}".format(
                str(init_channels), str(layers), str(w_lr), str(w_momentum), str(w_weight_decay),
                str(w_lr_step), str(datset_split), str(warm_up_epochs),
                str(pruning_step), str(gamma)))
    w_lr_min = 0.0001
    torch.cuda.set_device(0)
    seed = 2
    deterministic = False
    # torch.backends.cudnn.benchmark = True
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
    else:
        torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = get_data.get_data(
        'cifar10', '/userhome/temp_data/cifar10', cutout_length=0, validation=False,
        image_size=None)
    minimum_image_size = 32
    assert input_size >= minimum_image_size, "input image too small!!"

    # init model and net crit
    net_crit = nn.CrossEntropyLoss().to(device)
    from nas_201_api import NASBench201API as API
    api = API('/userhome/data/AutoML/NAS-Bench-102-v1_0-e61699.pth')
    model = NASBench201CNN(init_channels, layers, 4, n_classes, 'nas_bench_201')
    total_edges = model.num_edges
    num_ops = len(genotypes.NAS_BENCH_201)
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weight_parameters(), w_lr, momentum=w_momentum,
                              weight_decay=w_weight_decay)
    # split data to train/validation
    n_train = len(train_data)
    split = n_train - int(n_train / datset_split)
    indices = list(range(n_train))
    # shuffle data
    np.random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=256,
                                               sampler=train_sampler,
                                               num_workers=4,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=256,
                                               sampler=valid_sampler,
                                               num_workers=4,
                                               pin_memory=True)
    distribution_optimizer = Category_Dynamic_SNG_V3.Dynamic_SNG(categories=[num_ops] * total_edges,
                                                                 step=pruning_step,
                                                                 pruning=True, sample_with_prob=False,
                                                                 utility_function='log', utility_function_hyper=0.4,
                                                                 momentum=True, gamma=gamma)
    # training loop
    # step
    best_test = 0
    w_lr_step = w_lr_step * (num_ops / 8.) * (pruning_step / 3)
    # for epoch in range(warm_up_epochs):
    #     # lr_scheduler.step()
    #     lr = w_optim.param_groups[0]['lr']
    #     # warm up training
    #     array_sample = [random.sample(list(range(num_ops)), num_ops) for i in range(total_edges)]
    #     array_sample = np.array(array_sample)
    #     for i in range(num_ops):
    #         sample = np.transpose(array_sample[:, i])
    #         train(train_loader, valid_loader, model, w_optim, lr, epoch, sample, net_crit)
    best_top1 = 0.
    best_genotype = None
    lr_flag = 1
    for epoch in range(1000):
        if hasattr(distribution_optimizer, 'training_finish'):
            if distribution_optimizer.training_finish:
                break
        lr = w_optim.param_groups[0]['lr']
        sample = distribution_optimizer.sampling_index()

        # training
        train(train_loader, valid_loader, model, w_optim, lr, epoch, sample, net_crit)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step, sample, net_crit)
        # information recoder
        if lr > w_lr_min:
            if epoch >= lr_flag * w_lr_step and len(distribution_optimizer.sample_index[0]) == 0:
                utils.step_learning_rate(w_optim)
                lr_flag += 1
        distribution_optimizer.record_information(sample, top1)
        distribution_optimizer.update()
        # log
        # genotype
        genotype = model.genotype(distribution_optimizer.p_model.theta)
        # logger.info("the theta is = {}".format(distribution_optimizer.p_model.theta))

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
            index = api.query_index_by_arch(best_genotype)
            if index > 0:
                info = api.arch2infos_full[index].get_metrics('cifar10', 'ori-test')
                best_test = float(info['accuracy'])
        else:
            is_best = False
    genotype = model.genotype(distribution_optimizer.p_model.theta)
    index = api.query_index_by_arch(genotype)
    info = api.arch2infos_full[index].get_metrics('cifar10', 'ori-test')
    best_test = float(info['accuracy'])
    return best_test


def train(train_loader, valid_loader, model, w_optim, lr, epoch, sample, net_crit):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.to(device), trn_y.to(device)
        N = trn_X.size(0)
        w_optim.zero_grad()
        logits = model(trn_X, sample)
        loss = net_crit(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), 5.)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)


def validate(valid_loader, model, epoch, cur_step, sample, net_crit):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    # using model train instead
    # model.eval()
    model.train()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device), y.to(device)
            N = X.size(0)

            logits = model(X, sample)
            loss = net_crit(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
    return top1.avg


def array_main(param):
    result = []
    for i in range(4):
        a = main(init_channels=param['init_channels'], layers=param['layers'],
                 w_lr=param['w_lr'], w_momentum=param['w_momentum'], w_weight_decay=param['w_weight_decay'],
                 w_lr_step=param['w_lr_step'], datset_split=param['datset_split'], warm_up_epochs=param['warm_up_epochs'],
                 pruning_step=param['pruning_step'], gamma=param['gamma'])
        result.append(a)
    result_best = np.mean(np.array(result))
    result_best_var = np.var(np.array(result))
    logger.info("This trail best mean: {}, variance: {}".format(str(result_best), str(result_best_var)))
    return {'loss': -1 * result_best, 'status': STATUS_OK}


if __name__ == "__main__":

    # fspace = {
    #     'init_channels': hp.choice('init_channels', [8, 10, 12, 14, 16, 18, 20]),
    #     'layers': hp.choice('layers', [2, 3, 4, 5, 6, 7]),
    #     'w_lr': hp.choice('w_lr', [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]),
    #     'w_momentum': hp.choice('w_momentum', [0.5, 0.6, 0.7, 0.8, 0.9]),
    #     'w_weight_decay': hp.choice('w_weight_decay', [0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003]),
    #     'w_lr_step': hp.choice('w_lr_step', [5, 10, 15, 20, 25, 30, 35]),
    #     'datset_split': hp.choice('datset_split', [5, 10, 15, 20]),
    #     'warm_up_epochs': hp.choice('warm_up_epochs', [8, 10, 12, 14, 16, 18, 20]),
    #     'pruning_step': hp.choice('pruning_step', [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #     'gamma': hp.choice('gamma', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    # }
    fspace = {
        'init_channels': hp.choice('init_channels', [16]),
        'layers': hp.choice('layers', [5]),
        'w_lr': hp.choice('w_lr', [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]),
        'w_momentum': hp.choice('w_momentum', [0.1, 0.3, 0.5, 0.7, 0.9]),
        'w_weight_decay': hp.choice('w_weight_decay', [0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003]),
        'w_lr_step': hp.choice('w_lr_step', [5, 20, 25, 30]),
        'datset_split': hp.choice('datset_split', [10]),
        'warm_up_epochs': hp.choice('warm_up_epochs', [0, 1, 2, 3, 4, 5]),
        'pruning_step': hp.choice('pruning_step', [2, 3, 4]),
        'gamma': hp.choice('gamma', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    }
    # test function
    # example = {}
    # for key in fspace.keys():
    #     example[key] = hyperopt.pyll.stochastic.sample(fspace[key])
    # array_main(example)
    # pdb.set_trace()
    trails = Trials()
    best = fmin(fn=array_main, space=fspace, algo=tpe.suggest, max_evals=100, trials=trails, max_queue_len=4)
    logger.info("best: {}".format(str(best)))
    for trail in trails:
        logger.info("{}".format(trail))
    # main(init_channels=16, layers=5,
    #      w_lr=0.1, w_momentum=0.9, w_weight_decay=3e-4, w_lr_step=20,
    #      datset_split=10, warm_up_epochs=0,
    #      pruning_step=3, gamma=0.8)
