from xnas.search_space.cellbased_1shot1_ops import SearchSpace
import ConfigSpace
import numpy as np
import random
import os
import gc

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import xnas.core.meters as meters
import xnas.core.logging as logging
import xnas.core.config as config
import xnas.core.checkpoint as checkpoint
import xnas.core.distributed as dist

from xnas.core.utils import index_to_one_hot, one_hot_to_index
from xnas.core.trainer import setup_env
from xnas.core.timer import Timer
from xnas.core.config import cfg
from xnas.core.builders import build_space, build_loss_fun, lr_scheduler_builder, sng_builder

from xnas.datasets.loader import construct_loader
from xnas.nasbench.utils import EvaluateNasbench



# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()

# Tensorboard supplement
writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)


def random_sampling(search_space, distribution_optimizer, epoch=-1000, _random=False):
    """random sampling"""
    if _random:
        num_ops, total_edges = search_space.num_ops, search_space.all_edges
        # Edge importance
        non_edge_idx = []
        if cfg.SNG.EDGE_SAMPLING and epoch > cfg.SNG.EDGE_SAMPLING_EPOCH:
            assert cfg.SPACE.NAME in ['darts', 'nasbench301'], "only support darts for now!"
            norm_indexes = search_space.norm_node_index
            non_edge_idx = []
            for node in norm_indexes:
                # DARTS: N=7 nodes
                edge_non_prob = distribution_optimizer.p_model.theta[np.array(node), 7]
                edge_non_prob = edge_non_prob / np.sum(edge_non_prob)
                if len(node) == 2:
                    pass
                else:
                    non_edge_sampling_num = len(node) - 2
                    non_edge_idx += list(np.random.choice(node, non_edge_sampling_num, p=edge_non_prob, replace=False))
        # Big model sampling with probability
        if random.random() < cfg.SNG.BIGMODEL_SAMPLE_PROB:
            # Sample the network with high complexity
            _num = 100
            while _num > cfg.SNG.BIGMODEL_NON_PARA:
                _error = False
                if cfg.SNG.PROB_SAMPLING:
                    sample = np.array([np.random.choice(num_ops, 1, p=distribution_optimizer.p_model.theta[i, :])[0] for i in range(total_edges)])
                else:
                    sample = np.array([np.random.choice(num_ops, 1)[0] for i in range(total_edges)])
                _num = 0
                for i in sample[0:search_space.num_edges]:
                    if i in non_edge_idx:
                        pass
                    elif i in search_space.non_op_idx:
                        if i == 7:
                            _error = True
                        _num = _num + 1
                if _error:
                    _num = 100
        else:
            if cfg.SNG.PROB_SAMPLING:
                sample = np.array([np.random.choice(num_ops, 1, p=distribution_optimizer.p_model.theta[i, :])[0]
                                   for i in range(total_edges)])
            else:
                sample = np.array([np.random.choice(num_ops, 1)[0] for i in range(total_edges)])
        if cfg.SNG.EDGE_SAMPLING and epoch > cfg.SNG.EDGE_SAMPLING_EPOCH:
            for i in non_edge_idx:
                sample[i] = 7
        sample = index_to_one_hot(sample, distribution_optimizer.p_model.Cmax)
        # in the pruning method we have to sampling anyway
        distribution_optimizer.sampling()
        return sample
    else:
        return distribution_optimizer.sampling()


def train_model():
    """SNG search model training"""
    setup_env()
    # Load search space
    search_space = build_space()
    search_space.cuda()
    loss_fun = build_loss_fun().cuda()
    
    # Weights optimizer
    w_optim = torch.optim.SGD(search_space.parameters(),
                              cfg.OPTIM.BASE_LR,
                              momentum=cfg.OPTIM.MOMENTUM,
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    
    # Build distribution_optimizer
    if cfg.SPACE.NAME in ['darts', 'nasbench301']:
        distribution_optimizer = sng_builder([search_space.num_ops]*search_space.all_edges)
    elif cfg.SPACE.NAME in ['proxyless', 'google', 'ofa']:
        distribution_optimizer = sng_builder([search_space.num_ops]*search_space.all_edges)
    elif cfg.SPACE.NAME in ["nasbench1shot1_1", "nasbench1shot1_2", "nasbench1shot1_3"]:
        category = []
        cs = search_space.search_space.get_configuration_space()
        for h in cs.get_hyperparameters():
            if type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                category.append(len(h.choices))
        distribution_optimizer = sng_builder(category)
    else:
        raise NotImplementedError

    # Load dataset
    [train_, val_] = construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)

    lr_scheduler = lr_scheduler_builder(w_optim)
    all_timer = Timer()
    _over_all_epoch = 0

    # ===== Warm up training =====
    logger.info("start warm up training")
    warm_train_meter = meters.TrainMeter(len(train_))
    warm_val_meter = meters.TestMeter(len(val_))
    all_timer.tic()
    for cur_epoch in range(cfg.OPTIM.WARMUP_EPOCHS):
        
        # Save a checkpoint
        if (_over_all_epoch + 1) % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(
                search_space, w_optim, _over_all_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        
        lr = lr_scheduler.get_last_lr()[0]
        if cfg.SNG.WARMUP_RANDOM_SAMPLE:
            sample = random_sampling(search_space, distribution_optimizer, epoch=cur_epoch)
            logger.info("Sampling: {}".format(one_hot_to_index(sample)))
            train_epoch(train_, val_, search_space, w_optim, lr, _over_all_epoch, sample, loss_fun, warm_train_meter)
            top1 = test_epoch_with_sample(val_, search_space, warm_val_meter, _over_all_epoch, sample, writer)
            _over_all_epoch += 1
        else:
            num_ops, total_edges = search_space.num_ops, search_space.all_edges
            array_sample = [random.sample(list(range(num_ops)), num_ops) for i in range(total_edges)]
            array_sample = np.array(array_sample)
            for i in range(num_ops):
                sample = np.transpose(array_sample[:, i])
                sample = index_to_one_hot(sample, distribution_optimizer.p_model.Cmax)
                train_epoch(train_, val_, search_space, w_optim, lr, _over_all_epoch, sample, loss_fun, warm_train_meter)
                top1 = test_epoch_with_sample(val_, search_space, warm_val_meter, _over_all_epoch, sample, writer)
                _over_all_epoch += 1
    all_timer.toc()
    logger.info("end warm up training")
    
    # ===== Training procedure =====
    logger.info("start one-shot training")
    train_meter = meters.TrainMeter(len(train_))
    val_meter = meters.TestMeter(len(val_))
    all_timer.tic()
    for cur_epoch in range(cfg.OPTIM.MAX_EPOCH):
        
        # Save a checkpoint
        if (_over_all_epoch + 1) % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(
                search_space, w_optim, _over_all_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        
        if hasattr(distribution_optimizer, 'training_finish'):
            if distribution_optimizer.training_finish:
                break
        lr = w_optim.param_groups[0]['lr']

        sample = random_sampling(search_space, distribution_optimizer, epoch=cur_epoch, _random=cfg.SNG.RANDOM_SAMPLE)
        logger.info("Sampling: {}".format(one_hot_to_index(sample)))
        train_epoch(train_, val_, search_space, w_optim, lr, _over_all_epoch, sample, loss_fun, train_meter)
        top1 = test_epoch_with_sample(val_, search_space, val_meter, _over_all_epoch, sample, writer)
        _over_all_epoch += 1

        lr_scheduler.step()
        distribution_optimizer.record_information(sample, top1)
        distribution_optimizer.update()
    
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.SEARCH.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            logger.info("Start testing")
            logger.info("###############Optimal genotype at epoch: {}############".format(cur_epoch))
            logger.info(search_space.genotype(distribution_optimizer.p_model.theta))
            logger.info("########################################################")
            logger.info("####### ALPHA #######")
            for alpha in distribution_optimizer.p_model.theta:
                logger.info(alpha)
            logger.info("#####################")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
        gc.collect()
    all_timer.toc()

    # ===== Final epoch =====
    lr = w_optim.param_groups[0]['lr']
    all_timer.tic()
    for cur_epoch in range(cfg.OPTIM.FINAL_EPOCH):
        
        # Save a checkpoint
        if (_over_all_epoch + 1) % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(
                search_space, w_optim, _over_all_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        
        if cfg.SPACE.NAME in ['darts', 'nasbench301']:
            genotype = search_space.genotype(distribution_optimizer.p_model.theta)
            sample = search_space.genotype_to_onehot_sample(genotype)
        else:
            sample = distribution_optimizer.sampling_best()
        _over_all_epoch += 1
        train_epoch(train_, val_, search_space, w_optim, lr, _over_all_epoch, sample, loss_fun, train_meter)
        test_epoch_with_sample(val_, search_space, val_meter, _over_all_epoch, sample, writer)
    logger.info("Overall training time : {} hours".format(str((all_timer.total_time)/3600.)))

    # Evaluate through nasbench
    if cfg.SPACE.NAME in ["nasbench1shot1_1", "nasbench1shot1_2", "nasbench1shot1_3", "nasbench201", "nasbench301"]:
        logger.info("starting test using nasbench:{}".format(cfg.SPACE.NAME))
        theta = distribution_optimizer.p_model.theta
        EvaluateNasbench(theta, search_space, logger, cfg.SPACE.NAME)


def train_epoch(train_loader, valid_loader, model, w_optim, lr, cur_epoch, sample, net_crit, train_meter):
    model.train()
    train_meter.iter_tic()
    cur_step = cur_epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)
    # scale the grad in amp, amp only support the newest version
    scaler = torch.cuda.amp.GradScaler() if cfg.SEARCH.AMP & hasattr(
        torch.cuda.amp, 'autocast') else None
    
    for cur_iter, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda()
        # phase 1. child network step (w)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Perform the forward pass in AMP
                preds = model(trn_X, sample)
                # Compute the loss in AMP
                loss = net_crit(preds, trn_y)
                # Perform the backward pass in AMP
                w_optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(w_optim)
                # Updates the scale for next iteration.
                scaler.update()
        else:
            preds = model(trn_X, sample)
            # Compute the loss
            loss = net_crit(preds, trn_y)
            # Perform the backward pass
            w_optim.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
            # Update the parameters
            w_optim.step()
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


def test_epoch_with_sample(test_loader, model, test_meter, cur_epoch, sample, tensorboard_writer=None):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # using AMP
        if cfg.SEARCH.AMP & hasattr(torch.cuda.amp, 'autocast'):
            with torch.cuda.amp.autocast():
                # Compute the predictions
                preds = model(inputs, sample)
        else:
            # Compute the predictions
            preds = model(inputs, sample)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        # NOTE: this line is disabled before.
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    top1_err = test_meter.mb_top1_err.get_win_median()
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar(
            'val/top1_error', test_meter.mb_top1_err.get_win_median(), cur_epoch)
        tensorboard_writer.add_scalar(
            'val/top5_error', test_meter.mb_top5_err.get_win_median(), cur_epoch)
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()
    return top1_err


if __name__ == "__main__":
    train_model()
