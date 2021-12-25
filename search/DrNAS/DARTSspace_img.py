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
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy


from xnas.search_space.DrNAS.DARTSspace.cnn import NetworkImageNet as Network
import xnas.search_space.DrNAS.utils as utils


parser = argparse.ArgumentParser("imagenet")
parser.add_argument(
    "--workers", type=int, default=16, help="number of workers to load dataset"
)
parser.add_argument(
    "--data", type=str, default="datapath", help="location of the data corpus"
)
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.5, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.0, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument(
    "--init_channels", type=int, default=48, help="num of init channels"
)
parser.add_argument("--layers", type=int, default=14, help="total number of layers")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument("--save", type=str, default="exp", help="experiment name")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument(
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=6e-3,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay",
    type=float,
    default=1e-3,
    help="weight decay for arch encoding",
)
parser.add_argument("--k", type=int, default=6, help="init partial channel parameter")
parser.add_argument("--begin", type=int, default=10, help="warm start")

args = parser.parse_args()

args.save = "../experiments/imagenet/search-progressive-{}-{}-{}".format(
    args.save, time.strftime("%Y%m%d-%H%M%S"), args.seed
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

# data preparation, we random sample 10% and 2.5% from training set(each class) as train and val, respectively.
# Note that the data sampling can not use torch.utils.data.sampler.SubsetRandomSampler as imagenet is too large
CLASSES = 1000


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # dataset split
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    valid_data = dset.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    num_train = len(train_data)
    num_val = len(valid_data)
    print("# images to train network: %d" % num_train)
    print("# images to validate network: %d" % num_val)

    model = Network(args.init_channels, CLASSES, args.layers, criterion, k=args.k)
    model = nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer_a = torch.optim.Adam(
        model.module.arch_parameters(),
        lr=args.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay,
    )

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.workers,
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.workers,
    )

    # configure progressive parameter
    epoch = 0
    ks = [6, 3]
    num_keeps = [7, 4]
    train_epochs = [2, 2] if "debug" in args.save else [25, 25]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min
    )

    lr = args.learning_rate
    for i, current_epochs in enumerate(train_epochs):
        for e in range(current_epochs):
            current_lr = scheduler.get_lr()[0]
            logging.info("Epoch: %d lr: %e", epoch, current_lr)
            if epoch < 5 and args.batch_size > 256:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * (epoch + 1) / 5.0
                logging.info(
                    "Warming-up Epoch: %d, LR: %e", epoch, lr * (epoch + 1) / 5.0
                )
                print(optimizer)

            genotype = model.module.genotype()
            logging.info("genotype = %s", genotype)
            model.module.show_arch_parameters()

            epoch_start = time.time()
            # training
            train_acc, train_obj = train(
                train_queue, valid_queue, model, optimizer, optimizer_a, criterion, e
            )
            logging.info("Train_acc %f", train_acc)

            # validation
            if epoch >= 47:
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                logging.info("Valid_acc %f", valid_acc)
                # test_acc, test_obj = infer(test_queue, model, criterion)
                # logging.info('Test_acc %f', test_acc)

            epoch += 1
            scheduler.step()
            epoch_duration = time.time() - epoch_start
            logging.info("Epoch time: %ds.", epoch_duration)
            # utils.save(model, os.path.join(args.save, 'weights.pt'))

        if not i == len(train_epochs) - 1:
            model.module.pruning(num_keeps[i + 1])
            model.module.wider(ks[i + 1])
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

    genotype = model.module.genotype()
    logging.info("genotype = %s", genotype)
    model.module.show_arch_parameters()


def train(train_queue, valid_queue, model, optimizer, optimizer_a, criterion, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        if epoch >= args.begin:
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.sum().backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()
        # architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info(
                "TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
                duration,
            )
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
