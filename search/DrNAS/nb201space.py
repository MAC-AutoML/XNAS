import os
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

import xnas.core.config as config
import xnas.core.meters as meters
import xnas.core.logging as logging
import xnas.search_space.DrNAS.utils as utils
from xnas.search_algorithm.DrNAS import Architect
from xnas.core.builders import build_loss_fun, DrNAS_builder
from xnas.core.config import cfg
from xnas.core.timer import Timer
from xnas.core.trainer import setup_env, test_epoch
from xnas.datasets.loader import construct_loader

from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API


# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()
# Tensorboard supplement
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


# parser = argparse.ArgumentParser("sota")
# parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
# parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
# parser.add_argument('--method', type=str, default='dirichlet', help='choose nas method')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
# parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')

# NOTE: cutout is never used.
# parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
# parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')

# parser.add_argument('--save', type=str, default='exp', help='experiment name')
# parser.add_argument('--seed', type=int, default=2, help='random seed')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
# parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# parser.add_argument('--tau_max', type=float, default=10, help='Max temperature (tau) for the gumbel softmax.')
# parser.add_argument('--tau_min', type=float, default=1, help='Min temperature (tau) for the gumbel softmax.')
# parser.add_argument('--k', type=int, default=1, help='partial channel parameter')
#### regularization
# parser.add_argument('--reg_type', type=str, default='l2', choices=[
#                     'l2', 'kl'], help='regularization type, kl is implemented for dirichlet only')
# parser.add_argument('--reg_scale', type=float, default=1e-3,
#                     help='scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2')
# args = parser.parse_args()

# cfg.OUT_DIR = '../experiments/nasbench201/{}-search-{}-{}-{}'.format(
#     args.method, cfg.OUT_DIR, time.strftime("%Y%m%d-%H%M%S"), args.seed)
# if not args.dataset == 'cifar10':
#     cfg.OUT_DIR += '-' + args.dataset
# if args.unrolled:
#     cfg.OUT_DIR += '-unrolled'
# if not args.weight_decay == 3e-4:
#     cfg.OUT_DIR += '-weight_l2-' + str(args.weight_decay)
# if not args.arch_weight_decay == 1e-3:
#     cfg.OUT_DIR += '-alpha_l2-' + str(args.arch_weight_decay)
# if not args.method == 'gdas':
#     cfg.OUT_DIR += '-pc-' + str(args.k)

# utils.create_exp_dir(cfg.OUT_DIR, scripts_to_save=glob.glob('*.py'))

# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#     format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(cfg.OUT_DIR, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)
# writer = SummaryWriter(cfg.OUT_DIR + '/runs')


# if args.dataset == 'cifar100':
#     n_classes = 100
# elif args.dataset == 'imagenet16-120':
#     n_classes = 120
# else:
#     n_classes = 10


def distill(result):
    result = result.split("\n")
    cifar10 = result[5].replace(" ", "").split(":")
    cifar100 = result[7].replace(" ", "").split(":")
    imagenet16 = result[9].replace(" ", "").split(":")

    cifar10_train = float(cifar10[1].strip(",test")[-7:-2].strip("="))
    cifar10_test = float(cifar10[2][-7:-2].strip("="))
    cifar100_train = float(cifar100[1].strip(",valid")[-7:-2].strip("="))
    cifar100_valid = float(cifar100[2].strip(",test")[-7:-2].strip("="))
    cifar100_test = float(cifar100[3][-7:-2].strip("="))
    imagenet16_train = float(imagenet16[1].strip(",valid")[-7:-2].strip("="))
    imagenet16_valid = float(imagenet16[2].strip(",test")[-7:-2].strip("="))
    imagenet16_test = float(imagenet16[3][-7:-2].strip("="))

    return (
        cifar10_train,
        cifar10_test,
        cifar100_train,
        cifar100_valid,
        cifar100_test,
        imagenet16_train,
        imagenet16_valid,
        imagenet16_test,
    )


def main():

    setup_env()
    # follow DrNAS settings.
    torch.set_num_threads(3)
    cudnn.benchmark = True

    if not "debug" in cfg.OUT_DIR:
        api = API("./data/NAS-Bench-201-v1_1-096897.pth")

    criterion = build_loss_fun().cuda()
    # criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()

    assert cfg.DRNAS.METHOD in [
        "gdas",
        "snas",
        "dirichlet",
        "darts",
    ], "method not supported."

    if cfg.DRNAS.METHOD == "gdas" or cfg.DRNAS.METHOD == "snas":
        [tau_min, tau_max] = cfg.DRNAS.TAU
        # Create the decrease step for the gumbel softmax temperature
        tau_step = (tau_min - tau_max) / cfg.OPTIM.MAX_EPOCH
        tau_epoch = tau_max

    model = DrNAS_builder().cuda()

    # if args.method == 'gdas':
    #     model = TinyNetworkGDAS(C=cfg.SPACE.CHANNEL, N=cfg.SPACE.LAYERS, max_nodes=cfg.SPACE.NODES, num_classes=cfg.SEARCH.NUM_CLASSES,
    #                             criterion=criterion, search_space=NAS_BENCH_201)
    # elif args.method == 'snas':
    #     model = TinyNetwork(C=cfg.SPACE.CHANNEL, N=cfg.SPACE.LAYERS, max_nodes=cfg.SPACE.NODES, num_classes=cfg.SEARCH.NUM_CLASSES,
    #                         criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='gumbel',
    #                         reg_type="l2", reg_scale=1e-3)
    # elif args.method == 'dirichlet':
    #     model = TinyNetwork(C=cfg.SPACE.CHANNEL, N=cfg.SPACE.LAYERS, max_nodes=cfg.SPACE.NODES, num_classes=cfg.SEARCH.NUM_CLASSES,
    #                         criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='dirichlet',
    #                         reg_type=args.reg_type, reg_scale=args.reg_scale)
    # elif args.method == 'darts':
    #     model = TinyNetwork(C=cfg.SPACE.CHANNEL, N=cfg.SPACE.LAYERS, max_nodes=cfg.SPACE.NODES, num_classes=cfg.SEARCH.NUM_CLASSES,
    #                         criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='softmax',
    #                         reg_type="l2", reg_scale=1e-3)
    # model = model.cuda()
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.get_weights(),
        cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
    )

    # if args.dataset == 'cifar10':
    #     train_transform, valid_transform = utils._data_transforms_cifar10(args)
    #     train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    # elif args.dataset == 'cifar100':
    #     train_transform, valid_transform = utils._data_transforms_cifar100(args)
    #     train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    # elif args.dataset == 'svhn':
    #     train_transform, valid_transform = utils._data_transforms_svhn(args)
    #     train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
    # elif args.dataset == 'imagenet16-120':
    #     import torchvision.transforms as transforms
    #     from xnas.datasets.imagenet16 import ImageNet16
    #     mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    #     std = [x / 255 for x in [63.22,  61.26, 65.09]]
    #     lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
    #     train_transform = transforms.Compose(lists)
    #     train_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
    #     assert len(train_data) == 151700

    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))

    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=cfg.SEARCH.BATCH_SIZE,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    #     pin_memory=True)

    # valid_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=cfg.SEARCH.BATCH_SIZE,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    #     pin_memory=True)

    train_loader, valid_loader = construct_loader(
        cfg.SEARCH.DATASET,
        cfg.SEARCH.SPLIT,
        cfg.SEARCH.BATCH_SIZE,
        cfg.SEARCH.DATAPATH,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(cfg.OPTIM.MAX_EPOCH), eta_min=cfg.OPTIM.MIN_LR
    )

    architect = Architect(model, cfg)

    train_meter = meters.TrainMeter(len(train_loader))
    val_meter = meters.TestMeter(len(valid_loader))

    # train_timer = Timer()
    for current_epoch in range(cfg.OPTIM.MAX_EPOCH):
        lr = scheduler.get_lr()[0]
        logger.info("epoch %d lr %e", current_epoch, lr)

        genotype = model.genotype()
        logger.info("genotype = %s", genotype)
        model.show_arch_parameters(logger)

        # training
        # train_timer.tic()
        top1err = train_epoch(
            train_loader,
            valid_loader,
            model,
            architect,
            criterion,
            optimizer,
            lr,
            train_meter,
            current_epoch,
        )
        logger.info("Top1 err:%f", top1err)
        # train_timer.toc()
        # print("epoch time:{}".format(train_timer.diff))

        # validation
        # valid_acc, valid_obj = infer(valid_loader, model, criterion)
        # logger.info('valid_acc %f', valid_acc)
        test_epoch(valid_loader, model, val_meter, current_epoch, writer)

        if not "debug" in cfg.OUT_DIR:
            # nasbench201
            result = api.query_by_arch(model.genotype())
            logger.info("{:}".format(result))
            (
                cifar10_train,
                cifar10_test,
                cifar100_train,
                cifar100_valid,
                cifar100_test,
                imagenet16_train,
                imagenet16_valid,
                imagenet16_test,
            ) = distill(result)
            logger.info("cifar10 train %f test %f", cifar10_train, cifar10_test)
            logger.info(
                "cifar100 train %f valid %f test %f",
                cifar100_train,
                cifar100_valid,
                cifar100_test,
            )
            logger.info(
                "imagenet16 train %f valid %f test %f",
                imagenet16_train,
                imagenet16_valid,
                imagenet16_test,
            )

            # tensorboard
            # writer.add_scalars('accuracy', {'train':train_acc,'valid':valid_acc}, current_epoch)
            # writer.add_scalars('loss', {'train':train_obj,'valid':valid_obj}, current_epoch)
            writer.add_scalars(
                "nasbench201/cifar10",
                {"train": cifar10_train, "test": cifar10_test},
                current_epoch,
            )
            writer.add_scalars(
                "nasbench201/cifar100",
                {
                    "train": cifar100_train,
                    "valid": cifar100_valid,
                    "test": cifar100_test,
                },
                current_epoch,
            )
            writer.add_scalars(
                "nasbench201/imagenet16",
                {
                    "train": imagenet16_train,
                    "valid": imagenet16_valid,
                    "test": imagenet16_test,
                },
                current_epoch,
            )

            utils.save_checkpoint(
                {
                    "epoch": current_epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "alpha": model.arch_parameters(),
                },
                False,
                cfg.OUT_DIR,
            )

        scheduler.step()
        if cfg.DRNAS.METHOD == "gdas" or cfg.DRNAS.METHOD == "snas":
            # Decrease the temperature for the gumbel softmax linearly
            tau_epoch += tau_step
            logger.info("tau %f", tau_epoch)
            model.set_tau(tau_epoch)

        # print("avg epoch time:{}".format(train_timer.average_time))
        # train_timer.reset()

    writer.close()


def train_epoch(
    train_loader,
    valid_loader,
    model,
    architect,
    criterion,
    optimizer,
    lr,
    train_meter,
    cur_epoch,
):
    train_meter.iter_tic()
    cur_step = cur_epoch * len(train_loader)
    writer.add_scalar("train/lr", lr, cur_step)

    valid_loader_iter = iter(valid_loader)

    # objs = utils.AvgrageMeter()
    # top1 = utils.AvgrageMeter()
    # top5 = utils.AvgrageMeter()

    for cur_iter, (trn_X, trn_y) in enumerate(train_loader):
        model.train()
        try:
            (val_X, val_y) = next(valid_loader_iter)
        except StopIteration:
            valid_loader_iter = iter(valid_loader)
            (val_X, val_y) = next(valid_loader_iter)
        # Transfer the data to the current GPU device
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda(non_blocking=True)
        val_X, val_y = val_X.cuda(), val_y.cuda(non_blocking=True)

        # if epoch >= 15:
        architect.step(
            trn_X, trn_y, val_X, val_y, lr, optimizer, unrolled=cfg.DRNAS.UNROLLED
        )
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        logits = model(trn_X)
        loss = criterion(logits, trn_y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        top1_err, top5_err = meters.topk_errors(logits, trn_y, [1, 5])
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()

        # Update and log stats
        # TODO: multiply with NUM_GPUS are disabled before appling parallel
        # mb_size = trn_X.size(0) * cfg.NUM_GPUS
        mb_size = trn_X.size(0)
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        # write to tensorboard
        writer.add_scalar("train/loss", loss, cur_step)
        writer.add_scalar("train/top1_error", top1_err, cur_step)
        writer.add_scalar("train/top5_error", top5_err, cur_step)
        cur_step += 1
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return top1_err
    # if step % args.report_freq == 0:
    #     logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    #     if 'debug' in cfg.OUT_DIR:
    #         break

    # return  top1.avg, objs.avg


# def infer(valid_queue, model, criterion):
#     objs = utils.AvgrageMeter()
#     top1 = utils.AvgrageMeter()
#     top5 = utils.AvgrageMeter()
#     model.eval()

#     with torch.no_grad():
#         for step, (input, target) in enumerate(valid_queue):
#             input = input.cuda()
#             target = target.cuda(non_blocking=True)

#             logits = model(input)
#             loss = criterion(logits, target)

#             prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#             n = input.size(0)
#             objs.update(loss.data, n)
#             top1.update(prec1.data, n)
#             top5.update(prec5.data, n)

#             # if step % args.report_freq == 0:
#             #     logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
#             if 'debug' in cfg.OUT_DIR:
#                 break
#     return top1.avg, objs.avg


if __name__ == "__main__":
    main()
