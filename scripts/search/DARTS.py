"""DARTS searching"""

import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# DARTS
from xnas.algorithms.DARTS import *
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
    darts_controller = DartsCNNController(search_space, criterion).to(device)
    architect = Architect(darts_controller, cfg.OPTIM.MOMENTUM, cfg.OPTIM.WEIGHT_DECAY)
    
    # init optimizers
    w_optim = optimizer_builder("SGD", darts_controller.weights())
    a_optim = darts_alpha_optimizer("Adam", darts_controller.alphas())
    lr_scheduler = lr_scheduler_builder(w_optim)
    
    # init recorders
    darts_trainer = DartsTrainer(
        darts_controller=darts_controller,
        architect=architect,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        w_optim=w_optim,
        a_optim=a_optim,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    
    # load checkpoint or initial weights
    start_epoch = darts_trainer.darts_loading() if cfg.SEARCH.AUTO_RESUME else 0
    
    # start training
    darts_trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # train epoch
        darts_trainer.train_epoch(cur_epoch)
        # test epoch
        if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
            darts_trainer.test_epoch(cur_epoch)
            # recording genotype and alpha to logger
            logger.info("=== Optimal genotype at epoch: {} ===".format(cur_epoch))
            logger.info(darts_trainer.model.genotype())
            logger.info("=== alphas at epoch: {} ===".format(cur_epoch))
            darts_trainer.model.print_alphas(logger)
            # evaluate model
            if evaluator:
                evaluator(darts_trainer.model.genotype())
    darts_trainer.finish()


if __name__ == "__main__":
    main()
