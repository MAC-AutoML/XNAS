import os
import os
import time

import torch
import torch.backends.cudnn as cudnn

import xnas.core.config as config
import xnas.core.logging as logging
import xnas.search_space.DropNAS.utils as utils
from xnas.core.builders import build_space, build_loss_fun, lr_scheduler_builder
from xnas.core.config import cfg
from xnas.core.trainer import setup_env
from xnas.datasets.loader import construct_loader

# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()

# Tensorboard supplement
# writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def train(epoch, train_loader, model, criterion, optimizer):
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        choice = utils.random_choice(cfg.SPOS.NUMCHOICE, cfg.SPOS.LAYERS)
        outputs = model(inputs, choice)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        train_loss.update(loss.item(), n)
        train_acc.update(prec1.item(), n)
        if step % cfg.LOG_PERIOD == 0 or step == len(train_loader) - 1:
            logger.info(
                '[Supernet Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
                'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
                % (lr, epoch + 1, cfg.OPTIM.MAX_EPOCH, step + 1, steps_per_epoch,
                   loss.item(), train_loss.avg, prec1, train_acc.avg)
            )
    return train_loss.avg, train_acc.avg


def validate(val_loader, model, criterion):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            choice = utils.random_choice(cfg.SPOS.NUMCHOICE, cfg.SPOS.LAYERS)
            outputs = model(inputs, choice)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)
    return val_loss.avg, val_acc.avg


def main():
    setup_env()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    net = build_space()
    if torch.cuda.is_available():
        net = net.cuda()
        cudnn.benchmark = True

    [train_, val_] = construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE, cfg.SEARCH.DATAPATH)

    # Define Supernet
    net = net.cuda()
    loss_fun = build_loss_fun().cuda()
    optimizer = torch.optim.SGD(net.parameters(), cfg.OPTIM.BASE_LR, cfg.OPTIM.MOMENTUM, cfg.OPTIM.WEIGHT_DECAY)
    lr_scheduler = lr_scheduler_builder(optimizer)

    # Running
    start = time.time()
    best_val_acc = 0.0
    for epoch in range(cfg.OPTIM.MAX_EPOCH):
        # Supernet Training
        train_loss, train_acc = train(epoch, train_, net, loss_fun, optimizer)
        lr_scheduler.step()
        logger.info(
            '[Supernet Training] epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
            (epoch + 1, train_loss, train_acc)
        )
        # Supernet Validation
        val_loss, val_acc = validate(val_loader=val_, model=net, criterion=loss_fun)
        # Save Best Supernet Weights
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(cfg.OUT_DIR, '%s' % 'best.pth')
            torch.save(net.state_dict(), best_ckpt)
            logger.info('Save best checkpoints to %s' % best_ckpt)
        logger.info(
            '[Supernet Validation] epoch: %03d, val_loss: %.3f, val_acc: %.3f, best_acc: %.3f'
            % (epoch + 1, val_loss, val_acc, best_val_acc)
        )


if __name__ == '__main__':
    main()
