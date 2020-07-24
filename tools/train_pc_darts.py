import os
import gc
import sys
import time
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import xnas.core.checkpoint as checkpoint
import xnas.core.config as config
import xnas.core.logging as logging
import xnas.core.meters as meters
from xnas.core.builders import build_space
from xnas.core.config import cfg
from xnas.core.trainer import setup_env, test_epoch
from xnas.datasets.loader import _construct_loader
from xnas.search_algorithm.pc_darts import *
from xnas.search_space.cell_based import DartsCNN, NASBench201CNN
from torch.utils.tensorboard import SummaryWriter

# config load and assert
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()
# tensorboard
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def main():
    setup_env()
    # loadiong search space
    search_space = build_space()
    # init controller and architect
    loss_fun = nn.CrossEntropyLoss().cuda()
    darts_controller = PCDartsCNNController(search_space, loss_fun)
    darts_controller.cuda()
    architect = Architect(
        darts_controller, cfg.OPTIM.MOMENTUM, cfg.OPTIM.WEIGHT_DECAY)
    # load dataset
    [train_, val_] = _construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)
    # weights optimizer
    w_optim = torch.optim.SGD(darts_controller.weights(), cfg.OPTIM.BASE_LR, momentum=cfg.OPTIM.MOMENTUM,
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(darts_controller.alphas(), cfg.DARTS.ALPHA_LR, betas=(0.5, 0.999),
                                   weight_decay=cfg.DARTS.ALPHA_WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, cfg.OPTIM.MAX_EPOCH, eta_min=cfg.OPTIM.MIN_LR)
    train_meter = meters.TrainMeter(len(train_))
    val_meter = meters.TestMeter(len(val_))
    start_epoch = 0
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        lr = lr_scheduler.get_last_lr()[0]
        train_epoch(train_, val_, darts_controller, architect, loss_fun, w_optim, alpha_optim, lr, train_meter, cur_epoch)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(
                darts_controller, w_optim, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        lr_scheduler.step()
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.SEARCH.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            logger.info("Start testing")
            test_epoch(val_, darts_controller, val_meter, cur_epoch, tensorboard_writer=writer)
            logger.info("###############Optimal genotype at epoch: {}############".format(cur_epoch))
            logger.info(darts_controller.genotype())
            logger.info("########################################################")
            darts_controller.print_alphas(logger)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
        gc.collect()


def train_epoch(train_loader, valid_loader, model, architect, loss_fun, w_optimizer, alpha_optimizer, lr, train_meter, cur_epoch):
    model.train()
    train_meter.iter_tic()
    cur_step = cur_epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)
    # scale the grad in amp, amp only support the newest version
    scaler = torch.cuda.amp.GradScaler() if cfg.SEARCH.AMP & hasattr(
        torch.cuda.amp, 'autocast') else None
    valid_loader_iter = iter(valid_loader)
    for cur_iter, (trn_X, trn_y) in enumerate(train_loader):
        try:
            (val_X, val_y) = next(valid_loader_iter)
        except StopIteration:
            valid_loader_iter = iter(valid_loader)
            (val_X, val_y) = next(valid_loader_iter)
        # Transfer the data to the current GPU device
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda(non_blocking=True)
        val_X, val_y = val_X.cuda(), val_y.cuda(non_blocking=True)
        # phase 2. architect step (alpha)
        if cur_epoch >= 15:
            alpha_optimizer.zero_grad()
            architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optimizer)
            alpha_optimizer.step()

        # phase 1. child network step (w)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Perform the forward pass in AMP
                preds = model(trn_X)
                # Compute the loss in AMP
                loss = loss_fun(preds, trn_y)
                # Perform the backward pass in AMP
                w_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(w_optimizer)
                # Updates the scale for next iteration.
                scaler.update()
        else:
            preds = model(trn_X)
            # Compute the loss
            loss = loss_fun(preds, trn_y)
            # Perform the backward pass
            w_optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.weights(), cfg.OPTIM.GRAD_CLIP)
            # Update the parameters
            w_optimizer.step()
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, trn_y, [1, 5])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = trn_X.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        # write to tensorboard
        writer.add_scalar('train/loss', loss, cur_step)
        writer.add_scalar('train/top1_error', top1_err, cur_step)
        writer.add_scalar('train/top5_error', top5_err, cur_step)
        #print("###cur_iter:"+str(cur_iter)+",[loss,top1_error,top5_error]:"+str(loss)+","+str(top1_err)+","+str(top5_err))
        cur_step += 1
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


if __name__ == "__main__":
    main()
    writer.close()
