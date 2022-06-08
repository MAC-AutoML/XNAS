"""
    NBmacro: only (8 layers * 3 choices) + CIFAR10
"""

import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *
from xnas.runner.trainer import OneShotTrainer
from xnas.algorithms.SPOS import RAND, REA

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main():
    setup_env()
    criterion = criterion_builder().cuda()
    [train_loader, valid_loader] = construct_loader()
    model = space_builder().cuda()
    optimizer = optimizer_builder("SGD", model.parameters())
    lr_scheduler = lr_scheduler_builder(optimizer)

    # init sampler
    train_sampler = RAND(3, 8)
    evaluate_sampler = REA(3, 8)

    # init recorders
    nbm_trainer = OneShotTrainer(
        supernet=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        test_loader=valid_loader,
        sample_type='iter'
    )
    nbm_trainer.register_iter_sample(train_sampler)

    # load checkpoint or initial weights
    start_epoch = nbm_trainer.loading() if cfg.SEARCH.AUTO_RESUME else 0

    # start training
    nbm_trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # train epoch
        top1_err = nbm_trainer.train_epoch(cur_epoch)
        # test epoch
        if (cur_epoch + 1) % cfg.EVAL_PERIOD == 0 or (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH:
            top1_err = nbm_trainer.test_epoch(cur_epoch)
    nbm_trainer.finish()

    # # sample best architecture from supernet
    # for cycle in range(200):  # NOTE: this should be a hyperparameter
    #     sample = evaluate_sampler.suggest()
    #     top1_err = nbm_trainer.evaluate_epoch(sample)
    #     evaluate_sampler.record(sample, top1_err)
    # best_arch, best_top1err = evaluate_sampler.final_best()
    # logger.info("Best arch: {} \nTop1 error: {}".format(best_arch, best_top1err))

    from xnas.evaluations.NASBenchmacro.evaluate import Nbm_Eva
    # for example : arch = '00000000'
    arch = ''
    Nbm_Eva(arch)


if __name__ == '__main__':
    main()
