import os
import torch
import logging
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

import xnas.core.config as config
import xnas.core.meters as meters
import xnas.core.logging as logging
import xnas.search_space.DrNAS.utils as utils

from xnas.core.builders import build_loss_fun, DrNAS_builder
from xnas.core.config import cfg
from xnas.core.timer import Timer
from xnas.core.trainer import setup_env, test_epoch
from xnas.datasets.loader import construct_loader


# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()
# Tensorboard supplement
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)

# parser = argparse.ArgumentParser("imagenet")
# parser.add_argument(
#     "--workers", type=int, default=16, help="number of workers to load dataset"
# )
# parser.add_argument(
#     "--data", type=str, default="datapath", help="location of the data corpus"
# )
# parser.add_argument("--batch_size", type=int, default=512, help="batch size")
# parser.add_argument(
#     "--learning_rate", type=float, default=0.5, help="init learning rate"
# )
# parser.add_argument(
#     "--learning_rate_min", type=float, default=0.0, help="min learning rate"
# )
# parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
# parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
# parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
# parser.add_argument(
#     "--init_channels", type=int, default=48, help="num of init channels"
# )
# parser.add_argument("--layers", type=int, default=14, help="total number of layers")

# NOTE: cutout and drop_path_prob are never used.

# parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
# parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
# parser.add_argument(
#     "--drop_path_prob", type=float, default=0.3, help="drop path probability"
# )

# parser.add_argument("--save", type=str, default="exp", help="experiment name")
# parser.add_argument("--seed", type=int, default=0, help="random seed")
# parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
# parser.add_argument(
#     "--unrolled",
#     action="store_true",
#     default=False,
#     help="use one-step unrolled validation loss",
# )
# parser.add_argument(
#     "--arch_learning_rate",
#     type=float,
#     default=6e-3,
#     help="learning rate for arch encoding",
# )
# parser.add_argument(
#     "--arch_weight_decay",
#     type=float,
#     default=1e-3,
#     help="weight decay for arch encoding",
# )
# parser.add_argument("--k", type=int, default=6, help="init partial channel parameter")
# parser.add_argument("--begin", type=int, default=10, help="warm start")

# args = parser.parse_args()

# args.save = "../experiments/imagenet/search-progressive-{}-{}-{}".format(
#     args.save, time.strftime("%Y%m%d-%H%M%S"), args.seed
# )
# args.save += "-init_channels-" + str(args.init_channels)
# args.save += "-layers-" + str(args.layers)
# args.save += "-init_pc-" + str(args.k)
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))

# log_format = "%(asctime)s %(message)s"
# logging.basicConfig(
#     stream=sys.stdout,
#     level=logger.info,
#     format=log_format,
#     datefmt="%m/%d %I:%M:%S %p",
# )
# fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

# data preparation, we random sample 10% and 2.5% from training set(each class) as train and val, respectively.
# Note that the data sampling can not use torch.utils.data.sampler.SubsetRandomSampler as imagenet is too large


def data_preparation():
    traindir = os.path.join(cfg.SEAECH.DATAPATH, "train")
    valdir = os.path.join(cfg.SEAECH.DATAPATH, "val")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

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

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.SEARCH.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=cfg.SEARCH.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )

    return train_loader, valid_loader


def main():
    setup_env()
    cudnn.benchmark = True  # DrNAS code sets this term to True.

    criterion = build_loss_fun().cuda()
    # criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()

    model = DrNAS_builder()
    # model = Network(cfg.SPACE.CHANNEL, cfg.SEARCH.NUM_CLASSES, cfg.SPACE.LAYERS, criterion, k=cfg.DRNAS.K)
    model = nn.DataParallel(model)  # TODO: parallel not tested
    model = model.cuda()

    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
    )
    optimizer_a = torch.optim.Adam(
        model.module.arch_parameters(),
        lr=cfg.OPTIM.BASE_LR,
        betas=(0.5, 0.999),
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
    )

    [train_loader, valid_loader] = construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE, cfg.SEARCH.DATAPATH
    )
    # train_loader, valid_loader = data_preparation()

    # configure progressive parameter
    epoch = 0
    ks = [6, 3]
    num_keeps = [7, 4]
    train_epochs = [2, 2] if "debug" in cfg.OUT_DIR else [25, 25]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(sum(train_epochs)), eta_min=cfg.OPTIM.MIN_LR
    )

    lr = cfg.OPTIM.BASE_LR

    train_meter = meters.TrainMeter(len(train_loader))
    val_meter = meters.TestMeter(len(valid_loader))

    train_timer = Timer()
    for i, current_epochs in enumerate(train_epochs):
        for e in range(current_epochs):
            current_lr = scheduler.get_lr()[0]
            logger.info("Epoch: %d lr: %e", epoch, current_lr)
            if epoch < 5 and cfg.SEARCH.BATCH_SIZE > 256:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * (epoch + 1) / 5.0
                logger.info(
                    "Warming-up Epoch: %d, LR: %e", epoch, lr * (epoch + 1) / 5.0
                )
                print(optimizer)

            genotype = model.module.genotype()
            logger.info("genotype = %s", genotype)
            model.module.show_arch_parameters()

            train_timer.tic()
            # training
            top1err = train_epoch(
                train_loader,
                valid_loader,
                model,
                optimizer,
                optimizer_a,
                criterion,
                current_lr,
                train_meter,
                e,
            )
            logger.info("Top1 err:%f", top1err)

            train_timer.toc()
            print("epoch time:{}".format(train_timer.diff))

            # validation
            if epoch >= 47:
                # valid_acc, valid_obj = infer(valid_queue, model, criterion)
                # logger.info("Valid_acc %f", valid_acc)
                # test_acc, test_obj = infer(test_queue, model, criterion)
                # logger.info('Test_acc %f', test_acc)
                test_epoch(valid_loader, model, val_meter, epoch, writer)

            epoch += 1
            scheduler.step()

            if epoch % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
                save_ckpt(
                    model, os.path.join(cfg.OUT_DIR, "weights_epo" + str(epoch) + ".pt")
                )

        print("avg epoch time:{}".format(train_timer.average_time))
        train_timer.reset()

        if not i == len(train_epochs) - 1:
            model.module.pruning(num_keeps[i + 1])
            model.module.wider(ks[i + 1])
            optimizer = utils.configure_optimizer(
                optimizer,
                torch.optim.SGD(
                    model.parameters(),
                    cfg.OPTIM.BASE_LR,
                    momentum=cfg.OPTIM.MOMENTUM,
                    weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                ),
            )
            scheduler = utils.configure_scheduler(
                scheduler,
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(sum(train_epochs)), eta_min=cfg.OPTIM.MIN_LR
                ),
            )
            logger.info("pruning finish, %d ops left per edge", num_keeps[i + 1])
            logger.info("network wider finish, current pc parameter %d", ks[i + 1])

    genotype = model.module.genotype()
    logger.info("genotype = %s", genotype)
    model.module.show_arch_parameters()


def train_epoch(
    train_loader,
    valid_loader,
    model,
    optimizer,
    optimizer_a,
    criterion,
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

        if cur_epoch >= cfg.OPTIM.WARMUP_EPOCHS:
            optimizer_a.zero_grad()
            logits = model(val_X)
            loss_a = criterion(logits, val_y)
            loss_a.sum().backward()
            nn.utils.clip_grad_norm_(
                model.module.arch_parameters(), cfg.OPTIM.GRAD_CLIP
            )
            optimizer_a.step()
        # architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(trn_X)
        loss = criterion(logits, trn_y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()

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
    #     end_time = time.time()
    #     if step == 0:
    #         duration = 0
    #         start_time = time.time()
    #     else:
    #         duration = end_time - start_time
    #         start_time = time.time()
    #     logger.info(
    #         "TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds",
    #         step,
    #         objs.avg,
    #         top1.avg,
    #         top5.avg,
    #         duration,
    #     )
    # if "debug" in args.save:
    # break
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

#             if step % args.report_freq == 0:
#                 logger.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
#             if "debug" in args.save:
#                 break

#     return top1.avg, objs.avg


def save_ckpt(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_ckpt(model, model_path):
    model.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    main()
    writer.close()
