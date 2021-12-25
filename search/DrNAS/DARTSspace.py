from logging import critical
import os
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import xnas.core.config as config
import xnas.core.meters as meters
import xnas.core.logging as logging
import xnas.search_space.DrNAS.utils as utils
from xnas.search_algorithm.DrNAS import Architect

from xnas.core.builders import build_loss_fun, DrNAS_builder, lr_scheduler_builder
from xnas.core.config import cfg
from xnas.core.trainer import setup_env, test_epoch
from xnas.core.timer import Timer
from xnas.datasets.loader import construct_loader


# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()
# Tensorboard supplement
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


# parser = argparse.ArgumentParser("cifar")
# parser.add_argument(
#     "--data", type=str, default="datapath", help="location of the data corpus"
# )
# parser.add_argument(
#     "--dataset", type=str, default="cifar10", help="location of the data corpus"
# )
# parser.add_argument("--batch_size", type=int, default=64, help="batch size")
# parser.add_argument(
#     "--learning_rate", type=float, default=0.1, help="init learning rate"
# )
# parser.add_argument(
#     "--learning_rate_min", type=float, default=0.0, help="min learning rate"
# )
# parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
# parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
# parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
# parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
# parser.add_argument(
#     "--init_channels", type=int, default=36, help="num of init channels"
# )
# parser.add_argument("--layers", type=int, default=20, help="total number of layers")
# parser.add_argument("--save", type=str, default="exp", help="experiment name")
# parser.add_argument("--seed", type=int, default=2, help="random seed")
# parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
# parser.add_argument(
#     "--train_portion", type=float, default=0.5, help="portion of training data"
# )
# parser.add_argument(
#     "--unrolled",
#     action="store_true",
#     default=False,
#     help="use one-step unrolled validation loss",
# )
# parser.add_argument(
#     "--arch_learning_rate",
#     type=float,
#     default=6e-4,
#     help="learning rate for arch encoding",
# )
# parser.add_argument("--k", type=int, default=6, help="init partial channel parameter")
#### regularization
# parser.add_argument(
#     "--reg_type",
#     type=str,
#     default="l2",
#     choices=["l2", "kl"],
#     help="regularization type",
# )
# parser.add_argument(
#     "--reg_scale",
#     type=float,
#     default=1e-3,
#     help="scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2",
# )
# args = parser.parse_args()


# cfg.OUT_DIR = "../experiments/{}/search-progressive-{}-{}-{}".format(
#     cfg.SEARCH.DATASET, cfg.OUT_DIR, time.strftime("%Y%m%d-%H%M%S"), cfg.RNG_SEED
# )
# cfg.OUT_DIR += "-init_channels-" + str(cfg.SPACE.CHANNEL)
# cfg.OUT_DIR += "-layers-" + str(cfg.SPACE.LAYERS)
# cfg.OUT_DIR += "-init_pc-" + str(cfg.DRNAS.K)
# utils.create_exp_dir(cfg.OUT_DIR, scripts_to_save=glob.glob("*.py"))

# log_format = "%(asctime)s %(message)s"
# logging.basicConfig(
#     stream=sys.stdout,
#     level=logging.INFO,
#     format=log_format,
#     datefmt="%m/%d %I:%M:%S %p",
# )
# fh = logging.FileHandler(os.path.join(cfg.OUT_DIR, "log.txt"))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)


def main():

    setup_env()
    cudnn.benchmark = True  # DrNAS code sets this term to True.

    criterion = build_loss_fun().cuda()
    # criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()

    model = DrNAS_builder().cuda()
    # model = Network(
    #     cfg.SPACE.CHANNEL,
    #     cfg.SEARCH.NUM_CLASSES,
    #     cfg.SPACE.LAYERS,
    #     criterion,
    #     k=cfg.DRNAS.K,
    #     reg_type=cfg.DRNAS.REG_TYPE,
    #     reg_scale=cfg.DRNAS.REG_SCALE,
    # )
    model = model.cuda()
    architect = Architect(model, cfg)

    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
    )

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # if cfg.SEARCH.DATASET == "cifar100":
    #     train_data = dset.CIFAR100(
    #         root=cfg.SEARCH.DATAPATH, train=True, download=True, transform=train_transform
    #     )
    # else:
    #     train_data = dset.CIFAR10(
    #         root=cfg.SEARCH.DATAPATH, train=True, download=True, transform=train_transform
    #     )

    [train_queue, valid_queue] = construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE, cfg.SEARCH.DATAPATH
    )

    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))

    # train_queue = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=cfg.SEARCH.BATCH_SIZE,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    #     pin_memory=True,
    # )

    # valid_queue = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=cfg.SEARCH.BATCH_SIZE,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    #     pin_memory=True,
    # )

    # configure progressive parameter
    epoch = 0
    ks = [6, 4]
    num_keeps = [7, 4]
    train_epochs = [2, 2] if "debug" in cfg.OUT_DIR else [25, 25]

    lr_scheduler = lr_scheduler_builder(optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, float(sum(train_epochs)), eta_min=cfg.OPTIM.MIN_LR
    # )
    train_meter = meters.TrainMeter(len(train_queue))
    val_meter = meters.TestMeter(len(valid_queue))

    train_timer = Timer()
    for i, current_epochs in enumerate(train_epochs):
        train_timer.tic()
        logger.info("train period #{} total epochs {}".format(i, current_epochs))
        for e in range(current_epochs):
            lr = lr_scheduler.get_lr()[0]
            logger.info("epoch %d lr %e", epoch, lr)

            genotype = model.genotype()
            logger.info("genotype = %s", genotype)
            model.show_arch_parameters(logger)

            # training
            train_acc = train_epoch(
                train_queue,
                valid_queue,
                model,
                architect,
                criterion,
                optimizer,
                lr,
                train_meter,
                e,
            )
            logger.info("train_acc %f", train_acc)

            # validation
            # valid_acc, valid_obj = infer(valid_queue, model, criterion)
            # logger.info("valid_acc %f", valid_acc)
            test_epoch(valid_queue, model, val_meter, current_epochs, writer)

            epoch += 1
            lr_scheduler.step()

            if epoch % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
                save_ckpt(model, os.path.join(cfg.OUT_DIR, "weights.pt"))

        if not i == len(train_epochs) - 1:
            model.pruning(num_keeps[i + 1])
            # architect.pruning([model.mask_normal, model.mask_reduce])
            model.wider(ks[i + 1])
            optimizer = utils.configure_optimizer(
                optimizer,
                torch.optim.SGD(
                    model.parameters(),
                    cfg.OPTIM.BASE_LR,
                    momentum=cfg.OPTIM.MOMENTUM,
                    weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                ),
            )
            lr_scheduler = utils.configure_scheduler(
                lr_scheduler,
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(sum(train_epochs)), eta_min=cfg.OPTIM.MIN_LR
                ),
            )
            logger.info("pruning finish, %d ops left per edge", num_keeps[i + 1])
            logger.info("network wider finish, current pc parameter %d", ks[i + 1])

    genotype = model.genotype()
    logger.info("genotype = %s", genotype)
    model.show_arch_parameters(logger)


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

        if cur_epoch >= 10:
            architect.step(
                trn_X, trn_y, val_X, val_y, lr, optimizer, unrolled=cfg.DARTS.UNROLLED
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

    # prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
    # objs.update(loss.data, n)
    # top1.update(prec1.data, n)
    # top5.update(prec5.data, n)

    # if step % cfg.SEARCH.EVAL_PERIOD == 0:
    #     logger.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
    # if "debug" in cfg.OUT_DIR:
    #     break

    # return top1.avg, objs.avg


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

#             if step % cfg.SEARCH.EVAL_PERIOD == 0:
#                 logger.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
#             if "debug" in cfg.OUT_DIR:
#                 break

#     return top1.avg, objs.avg


def save_ckpt(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_ckpt(model, model_path):
    model.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    main()
    writer.close()
