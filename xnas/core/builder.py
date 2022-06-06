"""Builder for spaces, algorithms and evaluation metrics.

NOTE:
    This builder contains basic builders only.
    For specific methods, please import in code manualy.
    e.g.
        `darts_alpha_optimizer` is manualy in code `scripts/search/DARTS.py`,
        as it is not an general builder function.

"""

import os
import torch
import random
import numpy as np

import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg

# Dataloader
from xnas.datasets.loader import construct_loader
# Optimizers, criterions and LR_schedulers
from xnas.runner.optimizer import optimizer_builder
from xnas.runner.criterion import criterion_builder
from xnas.runner.scheduler import lr_scheduler_builder


__all__ = [
    'construct_loader', 
    'optimizer_builder',
    'criterion_builder',
    'lr_scheduler_builder',
    'space_builder',
    'SNG_builder',
    'evaluator_builder',
    'setup_env',
]


# -------------------------------------------------------- #
# Search Spaces Builder
# -------------------------------------------------------- #

from xnas.spaces.DARTS.cnn import _DartsCNN, _infer_DartsCNN
from xnas.spaces.PDARTS.cnn import _PDartsCNN
from xnas.spaces.PCDARTS.cnn import _PCDartsCNN
from xnas.spaces.NASBench201.cnn import _NASBench201, _infer_NASBench201
from xnas.spaces.DrNAS.darts_cnn import _DrNAS_DARTS_CNN
from xnas.spaces.DrNAS.nb201_cnn import _DrNAS_nb201_CNN, _GDAS_nb201_CNN
from xnas.spaces.SPOS.cnn import _SPOS_CNN, _infer_SPOS_CNN
from xnas.spaces.DropNAS.cnn import _DropNASCNN
from xnas.spaces.OFA.MobileNetV3.ofa_cnn import _OFAMobileNetV3
from xnas.spaces.OFA.ProxylessNet.ofa_cnn import _OFAProxylessNASNet
from xnas.spaces.OFA.ResNets.ofa_cnn import _OFAResNet


SUPPORTED_SPACES = {
    "darts": _DartsCNN,
    "pdarts": _PDartsCNN,
    "pcdarts": _PCDartsCNN,
    "nasbench201": _NASBench201,
    "drnas_darts": _DrNAS_DARTS_CNN,
    "drnas_nb201": _DrNAS_nb201_CNN,
    "gdas_nb201": _GDAS_nb201_CNN,
    "dropnas": _DropNASCNN,
    "spos": _SPOS_CNN,
    "ofa_mbv3": _OFAMobileNetV3,
    "ofa_proxyless": _OFAProxylessNASNet,
    "ofa_resnet": _OFAResNet,
    # models for inference
    "infer_darts": _infer_DartsCNN,
    "infer_nb201": _infer_NASBench201,
    "infer_spos": _infer_SPOS_CNN,
}

def space_builder(**kwargs):
    err_str = "Model type '{}' not supported".format(cfg.SPACE.NAME)
    assert cfg.SPACE.NAME in SUPPORTED_SPACES.keys(), err_str
    return SUPPORTED_SPACES[cfg.SPACE.NAME](**kwargs)


# -------------------------------------------------------- #
# Series Search Algorithms Builder
# -------------------------------------------------------- #

# === SNG series ===
from xnas.algorithms.SNG.SNG import SNG, Dynamic_SNG
from xnas.algorithms.SNG.ASNG import ASNG, Dynamic_ASNG
from xnas.algorithms.SNG.DDPNAS import CategoricalDDPNAS
from xnas.algorithms.SNG.MDENAS import CategoricalMDENAS
from xnas.algorithms.SNG.MIGO import MIGO
from xnas.algorithms.SNG.GridSearch import GridSearch
from xnas.algorithms.SNG.RAND import RandomSample


def SNG_builder(category):
    if cfg.SNG.NAME == 'SNG':
        return SNG(category, lam=cfg.SNG.LAMBDA)
    elif cfg.SNG.NAME == 'ASNG':
        return ASNG(category, lam=cfg.SNG.LAMBDA)
    elif cfg.SNG.NAME == 'Dynamic_SNG':
        return Dynamic_SNG(category, step=cfg.SNG.PRUNING_STEP, pruning=cfg.SNG.PRUNING)
    elif cfg.SNG.NAME == 'Dynamic_ASNG':
        return Dynamic_ASNG(category, step=cfg.SNG.PRUNING_STEP, pruning=cfg.SNG.PRUNING)
    elif cfg.SNG.NAME == 'MDENAS':
        return CategoricalMDENAS(category, cfg.SNG.THETA_LR)
    elif cfg.SNG.NAME == 'DDPNAS':
        return CategoricalDDPNAS(category, cfg.SNG.PRUNING_STEP, theta_lr=cfg.SNG.THETA_LR, gamma=cfg.SNG.GAMMA)
    elif cfg.SNG.NAME == 'MIGO':
        return MIGO(categories=category,
                    step=cfg.SNG.PRUNING_STEP, lam=cfg.SNG.LAMBDA,
                    pruning=cfg.SNG.PRUNING, sample_with_prob=cfg.SNG.PROB_SAMPLING,
                    utility_function=cfg.SNG.UTILITY, utility_function_hyper=cfg.SNG.UTILITY_FACTOR,
                    momentum=cfg.SNG.MOMENTUM, gamma=cfg.SNG.GAMMA, sampling_number_per_edge=cfg.SNG.SAMPLING_PER_EDGE)
    elif cfg.SNG.NAME == 'GridSearch':
        return GridSearch(category)
    elif cfg.SNG.NAME == 'RAND':
        return RandomSample(category)
    else:
        raise NotImplementedError


# -------------------------------------------------------- #
# Evaluations Builder
# -------------------------------------------------------- #

SUPPORTED_EVALUATIONS = {
    "nasbench201": ["nasbench201", "drnas_nb201", "gdas_nb201"],
    "nasbench301": ["darts", "pdarts", "pcdarts", "drnas_darts", "dropnas"],
}

def evaluator_builder():
    """Evaluator builder.

    Returns:
        evaluate: a function for evaluation.
    """
    if cfg.SEARCH.EVALUATION:
        err_str = "Evaluation method '{}' not supported".format(cfg.SEARCH.EVALUATION)
        assert cfg.SEARCH.EVALUATION in SUPPORTED_EVALUATIONS.keys(), err_str
        err_str = "Space '{}' is not supported by this evaluator".format(cfg.SPACE.NAME)
        assert cfg.SPACE.NAME in SUPPORTED_EVALUATIONS[cfg.SEARCH.EVALUATION], err_str
        logger.info("Evaluating with {}".format(cfg.SEARCH.EVALUATION))
        # import used evaluator only
        if cfg.SEARCH.EVALUATION == "nasbench201":
            import xnas.evaluations.NASBench201 as nb201
            return nb201.evaluate
        elif cfg.SEARCH.EVALUATION == "nasbench301":
            import xnas.evaluations.NASBench301 as nb301
            return nb301.evaluate
    return None



# -------------------------------------------------------- #

logger = logging.get_logger(__name__)

def setup_env():
    """Set up environment for training or testing."""
    # Ensure the output dir exists and save config
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    config.dump_cfgfile()

    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    if cfg.DETERMINSTIC:
        # Fix RNG seeds
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        # Configure the CUDNN backend
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCH

