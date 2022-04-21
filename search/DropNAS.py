import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import xnas.core.config as config
import xnas.core.logging as logging
from xnas.core.builders import build_loss_fun, lr_scheduler_builder
from xnas.core.config import cfg
from xnas.core.trainer import setup_env
from xnas.datasets.loader import get_data
from xnas.search_algorithm.DropNAS import SearchCNNController


from xnas.search_space.DropNAS import utils

config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()


# tensorboard
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def main():
    # logger.info("Logger is set - training start")


    setup_env()
    # set default gpu device id
    # torch.cuda.set_device(config.gpus[0])
    # search_space = build_space()
    loss_fun = build_loss_fun().cuda()

    # get data with meta info
    input_size, input_channels, n_classes, train_data = get_data(
        cfg.SEARCH.DATASET, cfg.SEARCH.DATAPATH, cutout_length=cfg.TRAIN.CUTOUT_LENGTH, validation=False)

    model = SearchCNNController(input_size, C_in=cfg.SEARCH.INPUT_CHANNEL, C=cfg.SPACE.CHANNEL,
                                n_classes=n_classes, n_layers=cfg.SPACE.LAYERS, criterion=loss_fun)
    model = model.cuda()

    # weights optimizer, weight decay is computed later in `train()`
    w_optim = torch.optim.SGD(model.weights(), cfg.OPTIM.BASE_LR, momentum=cfg.OPTIM.MOMENTUM,
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), cfg.DROPNAS.ALPHA_LR, betas=(0.5, 0.999), weight_decay=cfg.DROPNAS.ALPHA_WEIGHT_DECAY)

    # dataloader, we use the whole training data to search
    n_train = len(train_data)
    indices = list(range(n_train))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=cfg.SEARCH.BATCH_SIZE,
                                               sampler=train_sampler,
                                               num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                               pin_memory=True)
    #
    # [train_, val_] = construct_loader(
    #     cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE, cfg.SEARCH.DATAPATH)


    lr_scheduler = lr_scheduler_builder(w_optim)

    # train_meter = meters.TrainMeter(n_train)

    # training loop
    for epoch in range(cfg.OPTIM.MAX_EPOCH):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        drop_rate = 0. if epoch < cfg.DROPNAS.WARMUP_EPOCHS else cfg.DROPNAS.DROP_RATE
        logger.info("Current drop rate: {:.6f}".format(drop_rate))
        model.print_alphas(logger)

        # training
        train(train_loader, model, w_optim, alpha_optim, lr, epoch, drop_rate)

        # log genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))
        # with open(os.path.join(config.path, 'genotype.txt'), 'w') as f:
        #     f.write(str(genotype))
        #
        # utils.save_checkpoint(model, config.path, True)
        # print()


def train(train_loader, model, w_optim, alpha_optim, lr, epoch, drop_rate):

    #
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda()
        N = trn_X.size(0)

        # forward pass loss
        alpha_optim.zero_grad()
        w_optim.zero_grad()

        logits = model(trn_X, drop_rate=drop_rate)
        loss_1 = model.criterion(logits, trn_y)
        loss_1.backward()

        nn.utils.clip_grad_norm_(model.weights(), cfg.OPTIM.GRAD_CLIP)  # gradient clipping
        w_optim.step()
        if epoch >= cfg.DROPNAS.WARMUP_EPOCHS:
            alpha_optim.step()

        # weight decay loss
        loss_2 = model.weight_decay_loss(cfg.OPTIM.WEIGHT_DECAY) + model.alpha_decay_loss(cfg.DROPNAS.ALPHA_WEIGHT_DECAY)

        alpha_optim.zero_grad()
        w_optim.zero_grad()
        loss_2.backward()

        nn.utils.clip_grad_norm_(model.weights(), cfg.OPTIM.GRAD_CLIP)  # gradient clipping
        w_optim.step()
        alpha_optim.step()

        model.adjust_alphas()

        loss = loss_1 + loss_2

        # top1_err, top5_err = meters.topk_errors(logits, trn_y, [1, 5])

        # losses, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        # train_meter.iter_toc()
        # mb_size = trn_X.size(0)
        # train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        # train_meter.log_iter_stats(cur_epoch, cur_step)
        #
        # train_meter.iter_tic()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % cfg.TRAIN.CHECKPOINT_PERIOD == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, cfg.OPTIM.MAX_EPOCH, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, cfg.OPTIM.MAX_EPOCH, top1.avg))

if __name__ == "__main__":
    main()
    writer.close()