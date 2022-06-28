"""OFA evaluating.
    net config:
        {'ks': [5, 7, 7, 7, 7, 7, 7, 7, 5, 3, 7, 5, 3, 7, 5, 7, 3, 3, 3, 3], 
        'e': [3, 4, 6, 4, 4, 6, 6, 4, 3, 4, 3, 4, 4, 3, 4, 3, 6, 4, 3, 4], 
        'd': [2, 2, 2, 3, 3]
        }
"""

import os
import torch

import xnas.core.config as config
from xnas.datasets.loader import get_normal_dataloader
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# OFA
from xnas.spaces.OFA.utils import init_model, list_mean, set_running_statistics


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)
# Upper dir for supernet
upper_dir = os.path.join(*cfg.OUT_DIR.split('/')[:-1]) 



def main():
    setup_env()
    net = space_builder().cuda()
    # # [debug]
    # from xnas.spaces.OFA.MobileNetV3.ofa_cnn import _OFAMobileNetV3
    # net = _OFAMobileNetV3()
    checkpoint = torch.load(cfg.OFA.PATH, map_location="cpu")
    net.load_state_dict(checkpoint["model_state"])
    
    logger.info("load finished.")
    
    [train_loader, valid_loader] = get_normal_dataloader()
    
    test_meter = meter.TestMeter(len(valid_loader))
    
    
    # netcfg = net.sample_active_subnet()
    net.set_active_subnet(
        cfg.OFA.NETCFG['ks'],
        cfg.OFA.NETCFG['e'],
        cfg.OFA.NETCFG['d'],
    )
    
    set_running_statistics(net, valid_loader)
    net.eval()
    test_meter.reset(True)
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = net(inputs)
        top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()
        
        test_meter.iter_toc()
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        # test_meter.log_iter_stats(0, cur_iter)
        test_meter.iter_tic()
    top1_err = test_meter.mb_top1_err.get_global_avg()
    top5_err = test_meter.mb_top5_err.get_global_avg()
    # return top1_err, top5_err

    logger.info("-> top1_err:{} top5_err:{}".format(top1_err, top5_err))



if __name__ == '__main__':
    main()
