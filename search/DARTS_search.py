import gc
import os

from torch.utils.tensorboard import SummaryWriter

import xnas.core.benchmark as benchmark
import xnas.core.checkpoint as checkpoint
import xnas.core.config as config
import xnas.core.logging as logging
import xnas.core.meters as meters
from xnas.core.timer import Timer
from xnas.core.builders import build_space, build_loss_fun, lr_scheduler_builder
from xnas.core.config import cfg
from xnas.core.trainer import setup_env, test_epoch
from xnas.datasets.loader import construct_loader
from xnas.nasbench.utils import EvaluateNasbench
from xnas.search_algorithm.DARTS import *


# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()

# Tensorboard supplement
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def darts_train_model():
    """train DARTS model"""
    setup_env()
    # Loading search space
    search_space = build_space()
    # TODO: fix the complexity function
    # search_space = setup_model()
    # Init controller and architect
    loss_fun = build_loss_fun().cuda()
    darts_controller = DartsCNNController(search_space, loss_fun)
    darts_controller.cuda()
    architect = Architect(
        darts_controller, cfg.OPTIM.MOMENTUM, cfg.OPTIM.WEIGHT_DECAY)
    # Load dataset
    [train_, val_] = construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)
    # weights optimizer
    w_optim = torch.optim.SGD(darts_controller.weights(),
                              cfg.OPTIM.BASE_LR,
                              momentum=cfg.OPTIM.MOMENTUM,
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    # alphas optimizer
    a_optim = torch.optim.Adam(darts_controller.alphas(),
                               cfg.DARTS.ALPHA_LR,
                               betas=(0.5, 0.999),
                               weight_decay=cfg.DARTS.ALPHA_WEIGHT_DECAY)
    lr_scheduler = lr_scheduler_builder(w_optim)
    # Init meters
    train_meter = meters.TrainMeter(len(train_))
    val_meter = meters.TestMeter(len(val_))
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.SEARCH.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = darts_load_checkpoint(
            last_checkpoint, darts_controller, w_optim, a_optim)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.SEARCH.WEIGHTS:
        darts_load_checkpoint(cfg.SEARCH.WEIGHTS, darts_controller)
        logger.info("Loaded initial weights from: {}".format(
            cfg.SEARCH.WEIGHTS))
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(darts_controller, loss_fun, train_, val_)
    # Setup timer
    train_timer = Timer()
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    train_timer.tic()
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

        lr = lr_scheduler.get_last_lr()[0]
        train_epoch(train_, val_, darts_controller, architect,
                    loss_fun, w_optim, a_optim, lr, train_meter, cur_epoch)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
            checkpoint_file = darts_save_checkpoint(
                darts_controller, w_optim, a_optim, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        lr_scheduler.step()
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.SEARCH.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            logger.info("Start testing")
            test_epoch(val_, darts_controller, val_meter, cur_epoch, writer)
            logger.info(
                "###############Optimal genotype at epoch: {}############".format(cur_epoch))
            logger.info(darts_controller.genotype())
            logger.info(
                "########################################################")

            if cfg.SPACE.NAME == "nasbench301":
                logger.info("Evaluating with nasbench301")
                EvaluateNasbench(darts_controller.alpha, darts_controller.net, logger, "nasbench301")

            darts_controller.print_alphas(logger)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
        gc.collect()
    train_timer.toc()
    logger.info("Overall training time (hr) is:{}".format(
        str(train_timer.total_time)))


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
        alpha_optimizer.zero_grad()
        architect.unrolled_backward(
            trn_X, trn_y, val_X, val_y, lr, w_optimizer, unrolled=cfg.DARTS.SECOND)
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
        cur_step += 1
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def darts_save_checkpoint(model, w_optim, a_optim, epoch):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    # Ensure that the checkpoint dir exists
    os.makedirs(checkpoint.get_checkpoint_dir(), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state
    checkpoint_ = {
        "epoch": epoch,
        "model_state": sd,
        "w_optim_state": w_optim.state_dict(),
        "a_optim_state": a_optim.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = checkpoint.get_checkpoint(epoch + 1)
    torch.save(checkpoint_, checkpoint_file)
    return checkpoint_file


def darts_load_checkpoint(checkpoint_file, model, w_optim=None, a_optim=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint_ = torch.load(checkpoint_file, map_location="cpu")
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model.module if cfg.NUM_GPUS > 1 else model
    ms.load_state_dict(checkpoint_["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if w_optim:
        w_optim.load_state_dict(checkpoint_["w_optim_state"])
    if a_optim:
        a_optim.load_state_dict(checkpoint_["a_optim_state"])
    return checkpoint_["epoch"]

if __name__ == "__main__":
    darts_train_model()
    writer.close()
