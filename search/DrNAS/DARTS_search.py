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
from xnas.search_space.DrNAS.DARTSspace.cnn import NetworkCIFAR as Network
from xnas.search_algorithm.DrNAS import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="datapath", help="location of the data corpus"
)
parser.add_argument(
    "--dataset", type=str, default="cifar10", help="location of the data corpus"
)
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.0, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument(
    "--init_channels", type=int, default=36, help="num of init channels"
)
parser.add_argument("--layers", type=int, default=20, help="total number of layers")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument("--save", type=str, default="exp", help="experiment name")
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training data"
)
parser.add_argument(
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=6e-4,
    help="learning rate for arch encoding",
)
parser.add_argument("--k", type=int, default=6, help="init partial channel parameter")
#### regularization
parser.add_argument(
    "--reg_type",
    type=str,
    default="l2",
    choices=["l2", "kl"],
    help="regularization type",
)
parser.add_argument(
    "--reg_scale",
    type=float,
    default=1e-3,
    help="scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2",
)
args = parser.parse_args()

args.save = "../experiments/{}/search-progressive-{}-{}-{}".format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.seed
)
args.save += "-init_channels-" + str(args.init_channels)
args.save += "-layers-" + str(args.layers)
args.save += "-init_pc-" + str(args.k)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
if args.dataset == "cifar100":
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(
        args.init_channels,
        CIFAR_CLASSES,
        args.layers,
        criterion,
        k=args.k,
        reg_type=args.reg_type,
        reg_scale=args.reg_scale,
    )
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.dataset == "cifar100":
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform
        )
    else:
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform
        )

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
    )

    architect = Architect(model, args)

    # configure progressive parameter
    epoch = 0
    ks = [6, 4]
    num_keeps = [7, 4]
    train_epochs = [2, 2] if "debug" in args.save else [25, 25]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min
    )

    for i, current_epochs in enumerate(train_epochs):
        for e in range(current_epochs):
            lr = scheduler.get_lr()[0]
            logging.info("epoch %d lr %e", epoch, lr)

            genotype = model.genotype()
            logging.info("genotype = %s", genotype)
            model.show_arch_parameters()

            # training
            train_acc, train_obj = train(
                train_queue, valid_queue, model, architect, criterion, optimizer, lr, e
            )
            logging.info("train_acc %f", train_acc)

            # validation
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info("valid_acc %f", valid_acc)

            epoch += 1
            scheduler.step()
            utils.save(model, os.path.join(args.save, "weights.pt"))

        if not i == len(train_epochs) - 1:
            model.pruning(num_keeps[i + 1])
            # architect.pruning([model.mask_normal, model.mask_reduce])
            model.wider(ks[i + 1])
            optimizer = utils.configure_optimizer(
                optimizer,
                torch.optim.SGD(
                    model.parameters(),
                    args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                ),
            )
            scheduler = utils.configure_scheduler(
                scheduler,
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min
                ),
            )
            logging.info("pruning finish, %d ops left per edge", num_keeps[i + 1])
            logging.info("network wider finish, current pc parameter %d", ks[i + 1])

    genotype = model.genotype()
    logging.info("genotype = %s", genotype)
    model.show_arch_parameters()


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
            architect.step(
                input,
                target,
                input_search,
                target_search,
                lr,
                optimizer,
                unrolled=args.unrolled,
            )
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
            logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
        if "debug" in args.save:
            break

    return top1.avg, objs.avg


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
                logging.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
            if "debug" in args.save:
                break

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
