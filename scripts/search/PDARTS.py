"""PDARTS searching"""

from cmath import phase
import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# PDARTS
from xnas.algorithms.PDARTS import *
from xnas.runner.trainer import DartsTrainer
from xnas.runner.optimizer import darts_alpha_optimizer


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

def main():
    device = setup_env()
    criterion = criterion_builder().to(device)
    evaluator = evaluator_builder()

    [train_loader, valid_loader] = construct_loader()

    num_edges = (cfg.SPACE.NODES+3)*cfg.SPACE.NODES//2
    basic_op = []
    for i in range(num_edges * 2):
        basic_op.append(cfg.SPACE.BASIC_OP)

    for sp in range(len(cfg.PDARTS.NUM_TO_KEEP)):
        search_space = space_builder(
            add_layers=cfg.PDARTS.ADD_LAYERS[sp],
            add_width=cfg.PDARTS.ADD_WIDTH[sp],
            dropout_rate=float(cfg.PDARTS.DROPOUT_RATE[sp]),
            basic_op=basic_op,
        )

        # init models
        pdarts_controller = PDartsCNNController(search_space, criterion).to(device)
        architect = Architect(pdarts_controller, cfg.OPTIM.MOMENTUM, cfg.OPTIM.WEIGHT_DECAY)

        # init optimizers
        w_optim = optimizer_builder("SGD", pdarts_controller.subnet_weights())
        a_optim = darts_alpha_optimizer("Adam", pdarts_controller.alphas())
        lr_scheduler = lr_scheduler_builder(w_optim)

        # init recorders
        pdarts_trainer = DartsTrainer(
            darts_controller=pdarts_controller,
            architect=architect,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            w_optim=w_optim,
            a_optim=a_optim,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        # Load checkpoint or initial weights
        start_epoch = pdarts_trainer.darts_loading() if cfg.SEARCH.AUTO_RESUME else 0

        # start training
        pdarts_trainer.start()
        for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
            # train epoch
            if cur_epoch < cfg.PDARTS.EPS_NO_ARCHS[sp]:
                pdarts_trainer.model.update_p(
                    float(cfg.PDARTS.DROPOUT_RATE[sp]) *
                    (cfg.OPTIM.MAX_EPOCH - cur_epoch - 1) /
                    cfg.OPTIM.MAX_EPOCH
                )
                pdarts_trainer.train_epoch(cur_epoch, alpha_step=False)
            else:
                pdarts_trainer.model.update_p(
                    float(cfg.PDARTS.DROPOUT_RATE[sp]) * 
                    np.exp(-(cur_epoch - cfg.PDARTS.EPS_NO_ARCHS[sp]) * cfg.PDARTS.SCALE_FACTOR)
                )
                pdarts_trainer.train_epoch(cur_epoch, alpha_step=True)

            # test epoch
            if (cur_epoch+1) >= cfg.OPTIM.MAX_EPOCH - 5:
                pdarts_trainer.test_epoch(cur_epoch)
                # recording genotype and alpha to logger
                logger.info("=== Optimal genotype at epoch: {} ===".format(cur_epoch))
                logger.info(pdarts_trainer.model.genotype())
                logger.info("=== alphas at epoch: {} ===".format(cur_epoch))
                pdarts_trainer.model.print_alphas(logger)
                # evaluate model
                if evaluator:
                    evaluator(pdarts_trainer.model.genotype())
        pdarts_trainer.finish()
        logger.info("Top-{} primitive: {}".format(
            cfg.PDARTS.NUM_TO_KEEP[sp],
            pdarts_trainer.model.get_topk_op(cfg.PDARTS.NUM_TO_KEEP[sp]))
        )
        if sp == len(cfg.PDARTS.NUM_TO_KEEP) - 1:
            phase_ending(pdarts_trainer)
        else:
            basic_op = pdarts_trainer.model.get_topk_op(cfg.PDARTS.NUM_TO_KEEP[sp])
        phase_ending(pdarts_trainer)


def phase_ending(pdarts_trainer, final=False):
    phase = "Final" if final else "Stage"
    logger.info("=== {} optimal genotype ===".format(phase))
    logger.info(pdarts_trainer.model.genotype(final=True))
    logger.info("=== {} alphas ===".format(phase))
    pdarts_trainer.model.print_alphas(logger)
    # restrict skip connect
    logger.info('Restricting skip-connect')
    for sks in range(0, 9):
        max_sk = 8-sks
        num_sk = pdarts_trainer.model.get_skip_number()
        if not num_sk > max_sk:
            continue
        while num_sk > max_sk:
            pdarts_trainer.model.delete_skip()
            num_sk = pdarts_trainer.model.get_skip_number()
        
        logger.info('Number of skip-connect: %d', max_sk)
        logger.info(pdarts_trainer.model.genotype(final=True))


if __name__ == "__main__":
    main()
