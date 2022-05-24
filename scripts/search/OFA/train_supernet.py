"""OFA supernet training."""

import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# OFA
from xnas.runner.trainer import Trainer
from xnas.spaces.OFA.utils import init_model
from xnas.algorithms.OFA.progressive_shrinking import train_epoch, validate


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

def main():
    setup_env()
    # Network
    net = space_builder().cuda()
    init_model(net)
    # Loss function
    criterion = criterion_builder()
    # Data loaders
    [train_loader, valid_loader] = construct_loader()
    # Optimizers
    net_params = [
        # parameters with weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="exclude"), "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        # parameters without weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="include") , "weight_decay": 0}, 
    ]
    # init_lr = cfg.OPTIM.BASE_LR * cfg.NUM_GPUS    # TODO: multi-GPU support
    optimizer = optimizer_builder("SGD", net_params)
    lr_scheduler = lr_scheduler_builder(optimizer)
    
    # Initialize Recorder
    ofa_trainer = Trainer(
        model=net,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        test_loader=valid_loader,
    )
    # Resume
    start_epoch = ofa_trainer.loading() if cfg.SEARCH.AUTO_RESUME else 0

    # build validation config
    validate_func_dict = {
        "image_size_list": {cfg.TEST.IM_SIZE},
        "ks_list": sorted({min(net.ks_list), max(net.ks_list)}),
        "expand_ratio_list": sorted({min(net.expand_ratio_list), max(net.expand_ratio_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
    }
    if cfg.OFA.TASK == 'normal':
        pass
    elif cfg.OFA.TASK == 'kernel':
        validate_func_dict["ks_list"] = sorted(net.ks_list)
    elif cfg.OFA.TASK == 'depth':
        # add depth list constraints
        if (len(set(net.ks_list)) == 1) and (len(set(net.expand_ratio_list)) == 1):
            validate_func_dict["depth_list"] = net.depth_list
    elif cfg.OFA.TASK == 'expand':
        if len(set(net.ks_list)) == 1 and len(set(net.depth_list)) == 1:
            validate_func_dict["expand_ratio_list"] = net.expand_ratio_list
    else:
        raise NotImplementedError

    # Training
    logger.info("=== OFA | Task: {} | Phase: {} ===".format(cfg.OFA.TASK, cfg.OFA.PHASE))
    ofa_trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.WARMUP_EPOCH+cfg.OPTIM.MAX_EPOCH):
        train_epoch(
            train_=train_loader,
            net=net,
            train_criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            writer=ofa_trainer.writer,
            train_meter=ofa_trainer.train_meter,
            cur_epoch=cur_epoch,
        )
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            ofa_trainer.saving(cur_epoch)
        if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
            validate(
                val_=valid_loader,
                net=net,
                val_meter=ofa_trainer.test_meter,
                cur_epoch=cur_epoch,
                logger=logger,
                **validate_func_dict,
            )
            # TODO：OFA支持保存validation accuracy最高的checkpoint
    ofa_trainer.finish()


if __name__ == '__main__':
    main()
