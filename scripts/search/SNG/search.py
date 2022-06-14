"""SNG searching

(DARTS & nb201 space only)
"""

import torch
import random
import numpy as np

import xnas.core.config as config
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *
from xnas.core.utils import index_to_one_hot, one_hot_to_index

from xnas.runner.trainer import OneShotTrainer


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main():
    device = setup_env()
    search_space = space_builder().to(device)
    criterion = criterion_builder().to(device)
    evaluator = evaluator_builder()
    [train_loader, valid_loader] = construct_loader()
    
    w_optim = optimizer_builder("SGD", search_space.parameters())
    lr_scheduler = lr_scheduler_builder(w_optim)
    
    if cfg.SPACE.NAME in ['darts', 'nasbench201']:
        distribution_optimizer = SNG_builder([search_space.num_ops]*search_space.all_edges)
    else:
        raise NotImplementedError

    # Trainer definition
    sng_trainer = OneShotTrainer(
        supernet=search_space,
        criterion=criterion,
        optimizer=w_optim,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        test_loader=valid_loader,
    )
    
    over_all_epoch = 0
    
    # === Warmup ===
    logger.info("=== Warmup Training ===")
    sng_trainer.start()
    for cur_epoch in range(cfg.OPTIM.WARMUP_EPOCH):
        if cfg.SNG.WARMUP_RANDOM_SAMPLE:
            sample = random_sampling(search_space, distribution_optimizer, cur_epoch)
        else:
            num_ops, total_edges = search_space.num_ops, search_space.all_edges
            array_sample = [random.sample(list(range(num_ops)), num_ops) for i in range(total_edges)]
            array_sample = np.array(array_sample)
            for i in range(num_ops):
                sample = np.transpose(array_sample[:, i])
                sample = index_to_one_hot(sample, distribution_optimizer.p_model.Cmax)
        logger.info("Warmup Sampling: {}".format(one_hot_to_index(sample)))
        sng_trainer.train_epoch(over_all_epoch, sample)
        sng_trainer.test_epoch(over_all_epoch, sample)
        over_all_epoch += 1
    sng_trainer.finish()
    
    logger.info("=== Training ===")
    sng_trainer.start()
    for cur_epoch in range(cfg.OPTIM.MAX_EPOCH):
        if hasattr(distribution_optimizer, 'training_finish'):
            if distribution_optimizer.training_finish:
                break
        sample = random_sampling(search_space, distribution_optimizer, epoch=cur_epoch, _random=cfg.SNG.RANDOM_SAMPLE)
        logger.info("Sampling: {}".format(one_hot_to_index(sample)))
        sng_trainer.train_epoch(over_all_epoch, sample)
        top1_err = sng_trainer.test_epoch(over_all_epoch, sample)
        over_all_epoch += 1
        # TODO: REA & RAND in algorithm/SPOS are similar to this optimizer. Adding them to SNG series?
        distribution_optimizer.record_information(sample, top1_err)     
        distribution_optimizer.update()
        # Evaluate the model
        if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
            logger.info("=== Optimal genotype at epoch: {} ===".format(cur_epoch))
            geno = search_space.genotype(distribution_optimizer.p_model.theta)
            logger.info(geno)
            evaluator(geno, epoch=12)
            logger.info("=== alphas at epoch: {} ===".format(cur_epoch))
            for alpha in distribution_optimizer.p_model.theta:
                logger.info(alpha)
    sng_trainer.finish()
    
    logger.info("=== Final epochs ===")
    sng_trainer.start()
    for cur_epoch in range(cfg.OPTIM.FINAL_EPOCH):
        if cfg.SPACE.NAME in ['darts']:
            genotype = search_space.genotype(distribution_optimizer.p_model.theta)
            sample = search_space.genotype_to_onehot_sample(genotype)
        else:
            sample = distribution_optimizer.sampling_best()
        sng_trainer.train_epoch(over_all_epoch, sample)
        sng_trainer.test_epoch(over_all_epoch, sample)
        over_all_epoch += 1
    sng_trainer.finish()

    if cfg.SPACE.NAME in ['darts']:
        best_genotype = search_space.genotype(distribution_optimizer.p_model.theta)
        # evaluator(genotype)   # TODO: NAS-Bench-301 support.


def random_sampling(search_space, distribution_optimizer, epoch=-1000, _random=False):
    """random sampling"""
    if _random:
        num_ops, total_edges = search_space.num_ops, search_space.all_edges
        # Edge importance
        non_edge_idx = []
        if cfg.SNG.EDGE_SAMPLING and epoch > cfg.SNG.EDGE_SAMPLING_EPOCH:
            assert cfg.SPACE.NAME in ['darts'], "only support darts for now!"
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
                        if i == search_space.none_idx:
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
                sample[i] = search_space.none_idx
        sample = index_to_one_hot(sample, distribution_optimizer.p_model.Cmax)
        # in the pruning method we have to sampling anyway
        distribution_optimizer.sampling()
        return sample
    else:
        return distribution_optimizer.sampling()


if __name__ == "__main__":
    main()
