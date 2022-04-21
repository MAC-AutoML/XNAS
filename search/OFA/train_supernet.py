import xnas.core.logging as logging
import xnas.core.config as config
from xnas.core.config import cfg
import xnas.core.checkpoint as checkpoint
import xnas.core.benchmark as benchmark
from xnas.core.trainer import setup_env
from xnas.core.builders import build_space, build_loss_fun
from xnas.datasets.loader import construct_loader
from xnas.search_algorithm.OFA.progressive_shrinking import train 
import os
import torch
import torch.backends.cudnn as cudnn
from xnas.search_space.OFA.utils import init_models
from torch.utils.tensorboard import SummaryWriter


# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()

writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))
logger = logging.get_logger(__name__)




def main():
    setup_env()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    ### build net from cfg
    net = build_space()
    ### support cuda
    if torch.cuda.is_available():
        net = net.cuda()
        cudnn.benchmark = True
    ### init model
    init_models(net, 'he_fout')
    ### bulid loss function
    train_criterion = build_loss_fun()
    ### load dataset
    [train_, val_] = construct_loader(
        name=cfg.SEARCH.DATASET,
        split=cfg.SEARCH.SPLIT,
        batch_size=cfg.SEARCH.BATCH_SIZE,
        datapath=cfg.SEARCH.DATAPATH,
        backend=cfg.DATA_LOADER.BACKEND,
    )
    ### bulid optimizer
    net_params = [
        # parameters with weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="exclude"), "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        # parameters without weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="include") , "weight_decay": 0}, 
    ]
    init_lr = cfg.OPTIM.BASE_LR * cfg.NUM_GPUS
    optimizer = torch.optim.SGD(
        net_params,
        lr=init_lr,
        momentum=cfg.OPTIM.MOMENTUM,
        dampening=cfg.OPTIM.DAMPENING,
        nesterov=cfg.OPTIM.NESTEROV,
    )

    ### Load checkpoint or initial weights
    start_epoch = 0
    if cfg.SEARCH.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, net, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.SEARCH.WEIGHTS:
        checkpoint_epoch = checkpoint.load_checkpoint(cfg.SEARCH.WEIGHTS, net)
        logger.info("Loaded initial weights from: {}".format(cfg.SEARCH.WEIGHTS))


    """--------------------------------------------------------------------"""
    ### validation config
    validate_func_dict = {
        "image_size_list": {cfg.TEST.IM_SIZE},
        "ks_list": sorted({min(net.ks_list), max(net.ks_list)}),
        "expand_ratio_list": sorted({min(net.expand_ratio_list), max(net.expand_ratio_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
    }
    if cfg.OFA.TASK == 'normal':
        # 使用以上的默认设置即可
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
    
    logger.info('=========================== OFA | Task:{} | Phase:{} ==========================='.format(cfg.OFA.TASK, cfg.OFA.PHASE))
    
    
    """--------------------------------------------------------------------"""

    # TODO:
    # CIFAR10数据集，batch_size是64，compute_time_full函数会占7215MiB的显存，而正式训练和测试才只占900MiB
    # 发现是compute_time_train函数会占用很多显存，暂时未探究原因
    # ## Compute model and loader timings
    # if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
    #     benchmark.compute_time_full(net, train_criterion, train_, val_)

    ### Training
    train(
        net, start_epoch, optimizer, train_criterion, train_, val_,
        logger, writer, validate_func_dict,
    )



if __name__ == '__main__':
    main()
