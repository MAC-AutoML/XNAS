"""DropNAS searching"""

from torch import device
import torch.nn as nn

import xnas.core.config as config
import xnas.logger.logging as logging
import xnas.logger.meter as meter
from xnas.core.config import cfg
from xnas.core.builder import *

# DropNAS
from xnas.algorithms.DropNAS import *
from xnas.runner.trainer import DartsTrainer
from xnas.runner.optimizer import darts_alpha_optimizer


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

def main():
    device = setup_env()
    search_space = space_builder()
    criterion = criterion_builder().to(device)
    evaluator = evaluator_builder()
    
    [train_loader, valid_loader] = construct_loader()
    
    # init models
    darts_controller = DropNAS_CNNController(search_space, criterion).to(device)
    
    # init optimizers
    w_optim = optimizer_builder("SGD", darts_controller.weights())
    a_optim = darts_alpha_optimizer("Adam", darts_controller.alphas())
    lr_scheduler = lr_scheduler_builder(w_optim)
    
    # init recorders
    dropnas_trainer = DropNAS_Trainer(
        darts_controller=darts_controller,
        architect=None,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        w_optim=w_optim,
        a_optim=a_optim,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    
    # load checkpoint or initial weights
    start_epoch = dropnas_trainer.darts_loading() if cfg.SEARCH.AUTO_RESUME else 0
    
    # start training
    dropnas_trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # train epoch
        drop_rate = 0. if cur_epoch < cfg.OPTIM.WARMUP_EPOCH else cfg.DROPNAS.DROP_RATE
        logger.info("Current drop rate: {:.6f}".format(drop_rate))
        dropnas_trainer.train_epoch(cur_epoch, drop_rate)
        # test epoch
        if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
            # NOTE: the source code of DropNAS does not use test codes.
            # recording genotype and alpha to logger
            logger.info("=== Optimal genotype at epoch: {} ===".format(cur_epoch))
            logger.info(dropnas_trainer.model.genotype())
            logger.info("=== alphas at epoch: {} ===".format(cur_epoch))
            dropnas_trainer.model.print_alphas(logger)
            if evaluator:
                evaluator(dropnas_trainer.model.genotype())
    dropnas_trainer.finish()


class DropNAS_Trainer(DartsTrainer):
    """Trainer for DropNAS.
    Rewrite the train_epoch with DropNAS's double-losses policy.
    """
    def __init__(self, darts_controller, architect, criterion, w_optim, a_optim, lr_scheduler, train_loader, valid_loader):
        super().__init__(darts_controller, architect, criterion, w_optim, a_optim, lr_scheduler, train_loader, valid_loader)

    def train_epoch(self, cur_epoch, drop_rate):
        self.model.train()
        lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        for cur_iter, (trn_X, trn_y) in enumerate(self.train_loader):
            trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device, non_blocking=True)

            # forward pass loss
            self.a_optimizer.zero_grad()
            self.optimizer.zero_grad()
            preds = self.model(trn_X, drop_rate=drop_rate)
            loss1 = self.criterion(preds, trn_y)
            loss1.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()
            if cur_epoch >= cfg.OPTIM.WARMUP_EPOCH:
                self.a_optimizer.step()
            
            # weight decay loss
            self.a_optimizer.zero_grad()
            self.optimizer.zero_grad()
            loss2 = self.model.weight_decay_loss(cfg.OPTIM.WEIGHT_DECAY) \
                    + self.model.alpha_decay_loss(cfg.DARTS.ALPHA_WEIGHT_DECAY)
            loss2.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()
            self.a_optimizer.step()
            
            self.model.adjust_alphas()
            loss = loss1 + loss2

            # Compute the errors
            top1_err, top5_err = meter.topk_errors(preds, trn_y, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(top1_err, top5_err, loss, lr, trn_X.size(0))
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalar('train/loss', loss, cur_step)
            self.writer.add_scalar('train/top1_error', top1_err, cur_step)
            self.writer.add_scalar('train/top5_error', top5_err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        # saving model
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)


if __name__ == "__main__":
    main()
