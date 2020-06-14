""" Search cell """
import os
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


def mkdir(path):
    if os.path.isdir(path):
        return
    else:
        os.mkdir(path)


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(logdir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)
mkdir(config.path)
mkdir(config.network_info_path)

logger = utils.get_logger(os.path.join(config.path, "logger.log"))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")
    logger.info("Torch version is: {}".format(torch.__version__))

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    # torch.backends.cudnn.benchmark = True
    if config.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = get_data.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False,
        image_size=config.image_size)
    minimum_image_size = 4 if config.search_space == 'darts' else 32
    assert input_size >= minimum_image_size, "input image too small!!"

    # init model and net crit
    net_crit = nn.CrossEntropyLoss().to(device)
    if config.search_space == 'darts':
        model = SelectSearchCNN(input_channels, config.init_channels, n_classes,
                                config.layers, config.n_nodes, net_crit)
        total_edges = sum(list(range(2, config.n_nodes + 2))) * 2
        num_ops = len(genotypes.PRIMITIVES)
        model = model.to(device)
    elif config.search_space == 'nas_bench_201':
        from nas_201_api import NASBench201API as API
        api = API('/userhome/data/AutoML/NAS-Bench-102-v1_0-e61699.pth')
        model = NASBench201CNN(config.init_channels, config.layers, config.n_nodes, n_classes, config.search_space)
        total_edges = model.num_edges
        num_ops = len(genotypes.NAS_BENCH_201)
        model = model.to(device)
    elif config.search_space in ['proxyless', 'google', 'ofa']:
        model = get_super_net(n_classes=n_classes, base_stage_width=config.search_space,
                              width_mult=config.width_mult, conv_candidates=config.conv_candidates,
                              depth=config.depth)
        total_edges = len(model.blocks) - 1
        num_ops = len(config.conv_candidates) + 1
        model = model.to(device)
        super_net_config_path = os.path.join(config.network_info_path, 'supernet.json')
        super_net_config = model.config
        logger.info("Saving search supernet to {}".format(super_net_config_path))
        json.dump(super_net_config, open(super_net_config_path, 'a+'))
        flops_path = os.path.join(config.network_info_path, 'flops.json')
        flops_ = model.flops_counter_per_layer(input_size=[1, 3, 224, 224])
        logger.info("Saving flops to {}".format(flops_path))
        json.dump(flops_, open(flops_path, 'a+'))
    else:
        raise NotImplementedError

    # weights optimizer
    w_optim = torch.optim.SGD(model.weight_parameters(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # split data to train/validation
    n_train = len(train_data)
    split = n_train - int(n_train / config.datset_split)
    indices = list(range(n_train))
    # shuffle data
    np.random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)

    if config.name == 'MDENAS':
        distribution_optimizer = Category_MDENAS.CategoricalMDENAS(
            [num_ops]*total_edges, learning_rate=config.theta_lr)
    elif config.name == 'DDPNAS':
        distribution_optimizer = Category_DDPNAS.CategoricalDDPNAS(
            [num_ops]*total_edges, config.pruning_step)
    elif config.name == 'DDPNAS_V2':
        distribution_optimizer = Category_DDPNAS_V2.CategoricalDDPNASV2(
            [num_ops]*total_edges, config.pruning_step)
    elif config.name == 'DDPNAS_V3':
        distribution_optimizer = Category_DDPNAS_V3.CategoricalDDPNASV3(
            [num_ops] * total_edges, config.pruning_step, gamma=config.gamma)
    elif config.name == 'SNG':
        distribution_optimizer = Category_SNG.SNG(
            [num_ops]*total_edges)
    elif config.name == 'ASNG':
        distribution_optimizer = Category_ASNG.ASNG(
            [num_ops]*total_edges)
    elif config.name == 'dynamic_ASNG':
        distribution_optimizer = Category_Dynamic_ASNG.Dynamic_ASNG(categories=[num_ops]*total_edges,
                                                                    step=config.pruning_step,
                                                                    pruning=True)
    elif config.name == 'dynamic_SNG':
        distribution_optimizer = Category_Dynamic_SNG.Dynamic_SNG(categories=[num_ops]*total_edges,
                                                                  step=config.pruning_step,
                                                                  pruning=True)
    elif config.name == 'dynamic_SNG_V3':
        distribution_optimizer = Category_Dynamic_SNG_V3.Dynamic_SNG(categories=[num_ops]*total_edges,
                                                                     step=config.pruning_step,
                                                                     pruning=True, sample_with_prob=False,
                                                                     utility_function='log', utility_function_hyper=0.4,
                                                                     momentum=True, gamma=config.gamma)
    else:
        raise NotImplementedError
    # training loop
    # step
    config.w_lr_step = config.w_lr_step * (num_ops / 8.) * (config.pruning_step / 3)
    logger.info("learning rate step is: {}".format(str(config.w_lr_step)))

    logger.info("start warm up training")
    for epoch in range(config.warm_up_epochs):
        # lr_scheduler.step()
        lr = w_optim.param_groups[0]['lr']
        # warm up training
        array_sample = [random.sample(list(range(num_ops)), num_ops) for i in range(total_edges)]
        array_sample = np.array(array_sample)
        for i in range(num_ops):
            sample = np.transpose(array_sample[:, i])
            train(train_loader, valid_loader, model, w_optim, lr, epoch, sample, net_crit)
    logger.info("end warm up training")
    logger.info("start One shot searching")
    best_top1 = 0.
    best_genotype = None
    lr_flag = 1
    for epoch in range(config.epochs):
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
        if lr > config.w_lr_min:
            if 'dynamic' or 'DDPNAS' in config.name:
                if epoch >= lr_flag * config.w_lr_step and len(distribution_optimizer.sample_index[0]) == 0:
                    utils.step_learning_rate(w_optim)
                    lr_flag += 1
            else:
                if epoch % config.w_lr_step == 0 and epoch > 0:
                    utils.step_learning_rate(w_optim)
        distribution_optimizer.record_information(sample, top1)
        distribution_optimizer.update()
        # log
        # genotype
        genotype = model.genotype(distribution_optimizer.p_model.theta)
        logger.info("genotype: {}".format(genotype))
        logger.info("The learning rate is: {}".format(lr))
        # logger.info("the theta is = {}".format(distribution_optimizer.p_model.theta))

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
            logger.info("Current best genotype is: {}".format(genotype))
            if config.search_space == 'nas_bench_201':
                index = api.query_index_by_arch(best_genotype)
                if index > 0:
                    info = api.arch2infos_full[index].get_metrics('cifar10', 'ori-test')
                    logger.info('Test loss on CIFAR10 is: {}'.format(info['accuracy']))
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    logger.info("Training is done, saving the probability")
    np.save(os.path.join(config.network_info_path, 'probability.npy'), distribution_optimizer.p_model.theta)
    if config.search_space in ['proxyless', 'ofa', 'google']:
        logger.info("Generate the network config with 600M, 400M, 200M FLOPS")
        for i in [100, 200, 300, 400, 500, 600]:
            flops_save_name = '{0}_{1}_{2}_{3}_{4}'.format(config.name, config.search_space,
                                                           config.dataset, 'width_multi_' + str(config.width_mult),
                                                           str(i))
            path = get_MB_network(config.network_info_path, flops_constraint=i, name=flops_save_name)
            logger.info("FLOPS {}M: {}".format(str(i), str(path)))
    elif config.search_space == 'darts':
        for i in [2, 3, 4, 5, 6]:
            best_gene_constrain = get_gene_with_skip_connection_constraints(distribution_optimizer.p_model.theta,
                                                                            skip_constraint=i)
            logger.info("Best Genotype with N constraint {0} = {1}".format(str(i),best_gene_constrain))
        logger.info("Generate the network config with different constraints")
        get_gene_by_prob(config.network_info_path, distribution_optimizer.p_model.theta)
    logger.info("Done")


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
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


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

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
