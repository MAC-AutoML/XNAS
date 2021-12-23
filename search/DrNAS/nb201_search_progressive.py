import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import xnas.search_space.DrNAS.utils as utils
from xnas.search_space.DrNAS.nb201space.cnn import TinyNetwork
from xnas.search_space.DrNAS.nb201space.ops import NAS_BENCH_201
from xnas.search_algorithm.DrNAS import Architect


from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API


parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--method', type=str, default='dirichlet', help='choose nas method')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--tau_max', type=float, default=10, help='Max temperature (tau) for the gumbel softmax.')
parser.add_argument('--tau_min', type=float, default=1, help='Min temperature (tau) for the gumbel softmax.')
parser.add_argument('--k', type=int, default=4, help='init partial channel parameter')
#### regularization
parser.add_argument('--reg_type', type=str, default='l2', choices=[
                    'l2', 'kl'], help='regularization type, kl is implemented for dirichlet only')
parser.add_argument('--reg_scale', type=float, default=1e-3,
                    help='scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2')
args = parser.parse_args()

args.save = '../experiments/nasbench201/{}-search-progressive-{}-{}-{}'.format(
    args.method, args.save, time.strftime("%Y%m%d-%H%M%S"), args.seed)
if not args.dataset == 'cifar10':
    args.save += '-' + args.dataset
if args.unrolled:
    args.save += '-unrolled'
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
args.save += '-pc-' + str(args.k)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')


if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet16-120':
    n_classes = 120
else:
    n_classes = 10


def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    
    if not 'debug' in args.save:
        api = API('pth file path')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.method == 'snas':
        # Create the decrease step for the gumbel softmax temperature
        args.epochs = 100
        tau_step = (args.tau_min - args.tau_max) / args.epochs
        tau_epoch = args.tau_max
        model = TinyNetwork(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                            criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='gumbel')
    elif args.method == 'dirichlet':
        model = TinyNetwork(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                            criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='dirichlet', 
                            reg_type=args.reg_type, reg_scale=args.reg_scale)
    elif args.method == 'darts':
        model = TinyNetwork(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                            criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='softmax')
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.get_weights(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
    elif args.dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from xnas.datasets.imagenet16 import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    architect = Architect(model, args)
    
    # configure progressive parameter
    epoch = 0
    ks = [4, 2]
    num_keeps = [5, 3]
    train_epochs = [2, 2] if 'debug' in args.save else [50, 50]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min)
    
    for i, current_epochs in enumerate(train_epochs):
        for e in range(current_epochs):
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)
            model.show_arch_parameters()

            # training
            train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, e)
            logging.info('train_acc %f', train_acc)

            # validation
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

            if not 'debug' in args.save:
                # nasbench201
                result = api.query_by_arch(model.genotype())
                logging.info('{:}'.format(result))
                cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
                    cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)
                logging.info('cifar10 train %f test %f', cifar10_train, cifar10_test)
                logging.info('cifar100 train %f valid %f test %f', cifar100_train, cifar100_valid, cifar100_test)
                logging.info('imagenet16 train %f valid %f test %f', imagenet16_train, imagenet16_valid, imagenet16_test)

                # tensorboard
                writer.add_scalars('accuracy', {'train':train_acc,'valid':valid_acc}, epoch)
                writer.add_scalars('loss', {'train':train_obj,'valid':valid_obj}, epoch)
                writer.add_scalars('nasbench201/cifar10', {'train':cifar10_train,'test':cifar10_test}, epoch)
                writer.add_scalars('nasbench201/cifar100', {'train':cifar100_train,'valid':cifar100_valid, 'test':cifar100_test}, epoch)
                writer.add_scalars('nasbench201/imagenet16', {'train':imagenet16_train,'valid':imagenet16_valid, 'test':imagenet16_test}, epoch)

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'alpha': model.arch_parameters()
                }, False, args.save)
                
            epoch += 1
            scheduler.step()
            if args.method == 'snas':
                # Decrease the temperature for the gumbel softmax linearly
                tau_epoch += tau_step
                logging.info('tau %f', tau_epoch)
                model.set_tau(tau_epoch)

        if not i == len(train_epochs) - 1:
            model.pruning(num_keeps[i+1])
            # architect.pruning([model._mask])
            model.wider(ks[i+1])
            optimizer = utils.configure_optimizer(optimizer, torch.optim.SGD(
                model.get_weights(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay))
            scheduler = utils.configure_scheduler(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min))
            logging.info('pruning finish, %d ops left per edge', num_keeps[i+1])
            logging.info('network wider finish, current pc parameter %d', ks[i+1])

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)
    model.show_arch_parameters()
    writer.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)
        
        if epoch >= 10:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        if 'debug' in args.save:
            break

    return  top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
