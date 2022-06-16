"""Single Path One-Shot"""

import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# SPOS
from xnas.algorithms.SPOS import RAND, REA
from xnas.runner.trainer import OneShotTrainer


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

def main():
    device = setup_env()
    criterion = criterion_builder().to(device)
    [train_loader, valid_loader] = construct_loader()
    model = space_builder().cuda() #to(device)
    optimizer = optimizer_builder("SGD", model.parameters())
    lr_scheduler = lr_scheduler_builder(optimizer)
    
    # init sampler
    train_sampler = RAND(cfg.SPOS.NUM_CHOICE, cfg.SPOS.LAYERS)
    evaluate_sampler = REA(cfg.SPOS.NUM_CHOICE, cfg.SPOS.LAYERS)
    
    # init recorders
    spos_trainer = OneShotTrainer(
        supernet=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        test_loader=valid_loader,
        sample_type='iter'
    )
    spos_trainer.register_iter_sample(train_sampler)
    
    # load checkpoint or initial weights
    start_epoch = spos_trainer.loading() if cfg.SEARCH.AUTO_RESUME else 0
    
    # start training
    spos_trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # train epoch
        top1_err = spos_trainer.train_epoch(cur_epoch)
        # test epoch
        if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
            top1_err = spos_trainer.test_epoch(cur_epoch)
    spos_trainer.finish()
    
    # sample best architecture from supernet
    for cycle in range(200):    # NOTE: this should be a hyperparameter
        sample = evaluate_sampler.suggest()
        top1_err = spos_trainer.evaluate_epoch(sample)
        evaluate_sampler.record(sample, top1_err)
    best_arch, best_top1err = evaluate_sampler.final_best()
    logger.info("Best arch: {} \nTop1 error: {}".format(best_arch, best_top1err))

if __name__ == '__main__':
    main()
