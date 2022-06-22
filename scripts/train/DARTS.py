"""DARTS retraining"""
import torch
import torch.nn as nn

import xnas.core.config as config
import xnas.logger.logging as logging
import xnas.logger.meter as meter
from xnas.core.config import cfg
from xnas.core.builder import *
from xnas.datasets.loader import get_normal_dataloader

from xnas.runner.trainer import Trainer

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

def main():
    device = setup_env()
    
    model = space_builder().to(device)
    criterion = criterion_builder().to(device)
    # evaluator = evaluator_builder()
    
    [train_loader, valid_loader] = get_normal_dataloader()
    optimizer = optimizer_builder("SGD", model.parameters())
    lr_scheduler = lr_scheduler_builder(optimizer)
    
    darts_retrainer = Darts_Retrainer(
        model, criterion, optimizer, lr_scheduler, train_loader, valid_loader
    )
    
    start_epoch = darts_retrainer.loading() if cfg.SEARCH.AUTO_RESUME else 0
    
    darts_retrainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        darts_retrainer.model.drop_path_prob = cfg.TRAIN.DROP_PATH_PROB * cur_epoch / cfg.OPTIM.MAX_EPOCH
        darts_retrainer.train_epoch(cur_epoch)
        if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
            darts_retrainer.test_epoch(cur_epoch)
    darts_retrainer.finish()


# overwrite training & validating with auxiliary
class Darts_Retrainer(Trainer):
    def __init__(self, model, criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super().__init__(model, criterion, optimizer, lr_scheduler, train_loader, test_loader)
    
    def train_epoch(self, cur_epoch):
        self.model.train()
        lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device, non_blocking=True)
            preds, preds_aux = self.model(inputs)
            loss = self.criterion(preds, labels)
            self.optimizer.zero_grad()
            if cfg.TRAIN.AUX_WEIGHT > 0.:
                loss += cfg.TRAIN.AUX_WEIGHT * self.criterion(preds_aux, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()

            # Compute the errors
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS)
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalar('train/loss', loss, cur_step)
            self.writer.add_scalar('train/top1_error', top1_err, cur_step)
            self.writer.add_scalar('train/top5_error', top5_err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        self.lr_scheduler.step()
        # Saving checkpoint
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)
    
    @torch.no_grad()
    def test_epoch(self, cur_epoch):
        self.model.eval()
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device, non_blocking=True)
            preds, _ = self.model(inputs)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            
            self.test_meter.iter_toc()
            self.test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
            self.test_meter.log_iter_stats(cur_epoch, cur_iter)
            self.test_meter.iter_tic()
        top1_err = self.test_meter.mb_top1_err.get_win_avg()
        self.writer.add_scalar('val/top1_error', self.test_meter.mb_top1_err.get_win_avg(), cur_epoch)
        self.writer.add_scalar('val/top5_error', self.test_meter.mb_top5_err.get_win_avg(), cur_epoch)
        # Log epoch stats
        self.test_meter.log_epoch_stats(cur_epoch)
        self.test_meter.reset()
        # Saving best model
        if self.best_err > top1_err:
            self.best_err = top1_err
            self.saving(cur_epoch, best=True)

if __name__ == '__main__':
    main()
