"""DrNAS searching

The source code of DrNAS supports only cifar-10 & imagenet on DARTS space. 
To run on other datasets, please refer to:
    -> $XNAS/xnas/spaces/DrNAS/darts_cnn.py

DrNAS also provides support for GDAS and SNAS.

"""

import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# DARTS
from xnas.algorithms.DrNAS import Architect
from xnas.spaces.DrNAS.utils import *
from xnas.runner.trainer import DartsTrainer
from xnas.runner.optimizer import darts_alpha_optimizer


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

# DrNAS Hyperparameters initialization
tau_step, tau_epoch = None, None
ks, num_keeps, train_epochs = None, None, None
PRUNING_FLAG, TAU_FLAG, UNROLL_FLAG = True, False, False

def drnas_hp_builder():
    """Build hyper parameters for DrNAS"""
    global ks, num_keeps, train_epochs, tau_step, tau_epoch, \
           PRUNING_FLAG, TAU_FLAG, UNROLL_FLAG
    if cfg.SPACE.NAME == 'drnas_darts':
        assert cfg.LOADER.DATASET in ["cifar10", "cifar100", "imagenet16", "imagenet"]
        # darts space ignores the "cfg.DRNAS.METHOD"
        ks = [6, 3] if cfg.LOADER.DATASET == 'imagenet' else [6, 4]
        num_keeps = [7, 4]
        train_epochs = [25, 25]
    elif cfg.SPACE.NAME in ['drnas_nb201', 'gdas_nb201']:
        if cfg.DRNAS.PROGRESSIVE:
            assert cfg.DRNAS.METHOD in ["snas", "dirichlet", "darts"]
            ks = [4, 2]
            num_keeps = [5, 3]
            train_epochs = [50, 50]
        else:
            assert cfg.DRNAS.METHOD in ["gdas", "snas", "dirichlet", "darts"]
            PRUNING_FLAG = False
            UNROLL_FLAG = True
            train_epochs = [100]
        if cfg.DRNAS.METHOD in ["snas", "gdas"]:    # enable tau
            TAU_FLAG = True
            [tau_min, tau_max] = cfg.DRNAS.TAU
            # Create the decrease step for the gumbel softmax temperature
            tau_step = (tau_min - tau_max) / cfg.OPTIM.MAX_EPOCH
            tau_epoch = tau_max


def main():
    device = setup_env()
    criterion = criterion_builder()
    evaluator = evaluator_builder()
    
    [train_loader, valid_loader] = construct_loader()
    
    # DrNAS: configure progressive parameter
    drnas_hp_builder()
    
    # init models
    model = space_builder(criterion=criterion).to(device) # DrNAS combines the space with model controller.
    architect = None if cfg.LOADER.DATASET == 'imagenet' else Architect(model, cfg)
    
    # init optimizers
    w_optim = optimizer_builder("SGD", model.parameters())
    a_optim = darts_alpha_optimizer("Adam", model.arch_parameters()) \
        if cfg.LOADER.DATASET == 'imagenet' else None
    lr_scheduler = lr_scheduler_builder(w_optim, T_max=sum(train_epochs))
    
    # check whether warm-up training is used
    if cfg.LOADER.BATCH_SIZE <= 256 or cfg.LOADER.DATASET != 'imagenet':
        cfg.OPTIM.WARMUP_EPOCH = 0  # DrNAS does not warm-up if batch_size is small
    train_epochs[0] += cfg.OPTIM.WARMUP_EPOCH
    
    # init recorders
    drnas_trainer = DartsTrainer(
        darts_controller=model,
        architect=architect,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        w_optim=w_optim,
        a_optim=a_optim if cfg.LOADER.DATASET == 'imagenet' else architect.optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    
    # load checkpoint or initial weights
    start_epoch = drnas_trainer.darts_loading() if cfg.SEARCH.AUTO_RESUME else 0

    # start training
    drnas_trainer.start()
    for i, current_epoch in enumerate(train_epochs):
        if sum(train_epochs[:i+1]) <= start_epoch:
            continue
        for cur_epoch in range(start_epoch, sum(train_epochs[:i])+current_epoch):
            drnas_trainer.train_epoch(cur_epoch, (UNROLL_FLAG or cur_epoch>=10))
            if (cur_epoch+1) % cfg.EVAL_PERIOD == 0:
                logger.info("=== genotype at epoch: {} ===".format(cur_epoch))
                logger.info(drnas_trainer.model.genotype())
                logger.info("=== alphas at epoch: {} ===".format(cur_epoch))
                drnas_trainer.model.show_arch_parameters(logger)
                if cfg.LOADER.DATASET == 'imagenet' and cur_epoch < 47:
                    pass
                else:
                    drnas_trainer.test_epoch(cur_epoch)
                    if evaluator:
                        evaluator(
                            drnas_trainer.model.genotype(), 
                            writer=drnas_trainer.writer, 
                            cur_epoch=cur_epoch
                        )
            start_epoch += 1
        # set tau for snas & gdas
        if TAU_FLAG:
            tau_epoch += tau_step
            logger.info("tau %f", tau_epoch)
            drnas_trainer.model.set_tau(tau_epoch)
        # Pruning and transfer weights
        if PRUNING_FLAG and (not i == len(train_epochs) - 1):
            cfg.OPTIM.WARMUP_EPOCH = 0      # avoid warming-up again.
            drnas_trainer.model.pruning(num_keeps[i + 1])
            drnas_trainer.model.wider(ks[i + 1])
            drnas_trainer.optimizer = optimizer_transfer(
                drnas_trainer.optimizer,
                optimizer_builder("SGD", drnas_trainer.model.parameters())
            )
            drnas_trainer.lr_scheduler = scheduler_transfer(
                drnas_trainer.lr_scheduler, 
                lr_scheduler_builder(drnas_trainer.optimizer, T_max=sum(train_epochs))
            )
            logger.info(
                "Pruning finished. OPs per edge: {}, current pc parameter: {}"
                .format(num_keeps[i+1], ks[i+1])
            )
    logger.info("=== Final genotype ===")
    logger.info(drnas_trainer.model.genotype())
    logger.info("=== Final alphas ===")
    logger.info(drnas_trainer.model.show_arch_parameters())
    drnas_trainer.finish()


if __name__ == "__main__":
    main()
