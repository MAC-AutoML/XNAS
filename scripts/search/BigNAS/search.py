"""BigNAS subnet searching: Coarse-to-fine Architecture Selection"""

import numpy as np
from itertools import product

import torch

import xnas.core.config as config
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.builder import *
from xnas.core.config import cfg
from xnas.datasets.loader import get_normal_dataloader
from xnas.logger.meter import TestMeter


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def get_all_subnets():
    # get all subnets
    all_subnets = []
    subnet_sets = cfg.BIGNAS.SEARCH_CFG_SETS
    stage_names = ['mb1', 'mb2', 'mb3', 'mb4', 'mb5', 'mb6', 'mb7']

    mb_stage_subnets = []
    for mbstage in stage_names:
        mb_block_cfg = getattr(subnet_sets, mbstage)
        mb_stage_subnets.append(list(product(
            mb_block_cfg.c,
            mb_block_cfg.d,
            mb_block_cfg.k,
            mb_block_cfg.t
        )))

    all_mb_stage_subnets = list(product(*mb_stage_subnets))

    resolutions = getattr(subnet_sets, 'resolutions')
    first_conv = getattr(subnet_sets, 'first_conv')
    last_conv = getattr(subnet_sets, 'last_conv')

    for res in resolutions:
        for fc in first_conv.c:
            for mb in all_mb_stage_subnets:
                np_mb_choice = np.array(mb)
                width = np_mb_choice[:, 0].tolist()  # c
                depth = np_mb_choice[:, 1].tolist()  # d
                kernel = np_mb_choice[:, 2].tolist() # k
                expand = np_mb_choice[:, 3].tolist() # t
                for lc in last_conv.c:
                    all_subnets.append({
                        'resolution': res,
                        'width': [fc] + width + [lc],
                        'depth': depth,
                        'kernel_size': kernel,
                        'expand_ratio': expand
                    })
    return all_subnets


def main():
    setup_env()
    supernet = space_builder().cuda()
    supernet.load_weights_from_pretrained_models(cfg.SEARCH.WEIGHTS)

    [train_loader, valid_loader] = get_normal_dataloader()

    test_meter = TestMeter(len(valid_loader))

    all_subnets = get_all_subnets()
    benchmarks = []

    # Phase 1. coarse search
    for k,subnet_cfg in enumerate(all_subnets):
        supernet.set_active_subnet(
            subnet_cfg['resolution'],
            subnet_cfg['width'],
            subnet_cfg['depth'],
            subnet_cfg['kernel_size'],
            subnet_cfg['expand_ratio'],
        )
        subnet = supernet.get_active_subnet().cuda()
        
        # Validate
        top1_err, top5_err = validate(subnet, train_loader, valid_loader, test_meter)
        flops = supernet.compute_active_subnet_flops()

        logger.info("[{}/{}] flops:{} top1_err:{} top5_err:{}".format(
            k+1, len(all_subnets), flops, top1_err, top5_err
        ))

        benchmarks.append({
            'subnet_cfg': subnet_cfg,
            'flops': flops,
            'top1_err': top1_err,
            'top5_err': top5_err
        })

    # Phase 2. fine-grained search
    try:
        best_subnet_info = list(filter(
            lambda k: k['flops'] < cfg.BIGNAS.CONSTRAINT_FLOPS,
            sorted(benchmarks, key=lambda d: d['top1_err'])))[0]
        best_subnet_cfg = best_subnet_info['subnet_cfg']
        best_subnet_top1 = best_subnet_info['top1_err']
    except IndexError:
        logger.info("Cannot find subnets under {} FLOPs".format(cfg.BIGNAS.CONSTRAINT_FLOPS))
        exit(1)
    
    for mutate_epoch in range(cfg.BIGNAS.NUM_MUTATE):
        new_subnet_cfg = supernet.mutate_and_reset(best_subnet_cfg)
        prev_cfgs = [i['subnet_cfg'] for i in benchmarks]
        if new_subnet_cfg in prev_cfgs:
            continue
        
        subnet = supernet.get_active_subnet().cuda()
        # Validate
        top1_err, top5_err = validate(subnet, train_loader, valid_loader, test_meter)
        flops = supernet.compute_active_subnet_flops()
        
        logger.info("[{}/{}] flops:{} top1_err:{} top5_err:{}".format(
            mutate_epoch+1, cfg.BIGNAS.NUM_MUTATE, flops, top1_err, top5_err
        ))

        benchmarks.append({
            'subnet_cfg': subnet_cfg,
            'flops': flops,
            'top1_err': top1_err,
            'top5_err': top5_err
        })
        
        if flops < cfg.BIGNAS.CONSTRAINT_FLOPS and top1_err < best_subnet_top1:
            best_subnet_cfg = new_subnet_cfg
            best_subnet_top1 = top1_err
    
    # Final best architecture
    logger.info("="*20 + "\nMutate Finished.")
    logger.info("Best Architecture:\n{}\n Best top1_err:{}".format(
        best_subnet_cfg, best_subnet_top1
    ))


@torch.no_grad()
def validate(subnet, train_loader, valid_loader, test_meter):
    # BN calibration
    subnet.eval()
    logger.info("Calibrating BN running statistics.")
    subnet.reset_running_stats_for_calibration()
    for cur_iter, (inputs, _) in enumerate(train_loader):
        if cur_iter >= cfg.BIGNAS.POST_BN_CALIBRATION_BATCH_NUM:
            break
        inputs = inputs.cuda()
        subnet(inputs)      # forward only

    top1_err, top5_err = test_epoch(subnet, valid_loader, test_meter)
    return top1_err, top5_err


def test_epoch(subnet, test_loader, test_meter):
    subnet.eval()
    test_meter.reset(True)
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = subnet(inputs)
        top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()

        test_meter.iter_toc()
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(0, cur_iter)
        test_meter.iter_tic()
    top1_err = test_meter.mb_top1_err.get_win_avg()
    top5_err = test_meter.mb_top5_err.get_win_avg()
    # self.writer.add_scalar('val/top1_error', test_meter.mb_top1_err.get_win_avg(), cur_epoch)
    # self.writer.add_scalar('val/top5_error', test_meter.mb_top5_err.get_win_avg(), cur_epoch)
    # Log epoch stats
    test_meter.log_epoch_stats(0)
    # test_meter.reset()
    return top1_err, top5_err


if __name__ == "__main__":
    main()
