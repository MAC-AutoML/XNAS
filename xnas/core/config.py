#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys

from yacs.config import CfgNode as CfgNode

# Global config object
_C = CfgNode()
_C.IMMUTABLE=True
# Example usage:
#   from core.config import cfg
cfg = _C

_C.SPACE = CfgNode()

# ------------------------------------------------------------------------------------ #
# Search Space options
# ------------------------------------------------------------------------------------ #
_C.SPACE.NAME = 'darts'

# Loss function
_C.SPACE.LOSS_FUN = "cross_entropy"

# num of classes
_C.SPACE.NUM_CLASSES = 10

# Init channel
_C.SPACE.CHANNEL = 16

# number of layers
_C.SPACE.LAYERS = 8

# number of nodes in a cell
_C.SPACE.NODES = 4

#number of  PRIMITIVE
_C.SPACE.PRIMITIVES=[ ]

# number of nodes in a cell
_C.SPACE.BASIC_OP = []



# ------------------------------------------------------------------------------------ #
# Optimizer options in network
# ------------------------------------------------------------------------------------ #
_C.OPTIM = CfgNode()

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Exponential decay factor
_C.OPTIM.GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Minimal learning rate in cosine
_C.OPTIM.MIN_LR = 0.001

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# Momentum dampening
_C.OPTIM.GRAD_CLIP = 5.0


# ------------------------------------------------------------------------------------ #
# Common train/test data loader options
# ------------------------------------------------------------------------------------ #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True

# using which backend as image decoder and transformers: dali_cpu, dali_gpu, torch, and custom
_C.DATA_LOADER.BACKEND = 'dali_cpu'

# Number of data loader workers per process
_C.DATA_LOADER.WORLD_SIZE = 1

# Copy the whole dataset into memory
_C.DATA_LOADER.MEMORY_DATA = False

# transformers
_C.DATA_LOADER.PCA_JITTER = False
_C.DATA_LOADER.COLOR_JITTER = False

# ------------------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------------------ #
_C.SEARCH = CfgNode()

# Dataset and split
_C.SEARCH.DATASET = "cifar10"
_C.SEARCH.SPLIT = [0.8, 0.2]

# Total mini-batch size
_C.SEARCH.BATCH_SIZE = 256

# Image size
_C.SEARCH.IM_SIZE = 32

# Evaluate model on test data every eval period epochs
_C.SEARCH.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.SEARCH.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
_C.SEARCH.AUTO_RESUME = True

# Weights to start training from
_C.SEARCH.WEIGHTS = ""

# using FP16
_C.SEARCH.AMP = False

#adujust the number of layers
_C.SEARCH.add_layers=0

#adujust the width
_C.SEARCH.add_width=0

#droupout_rate of skip operation
_C.SEARCH.dropout_rate=0.0

# ------------------------------------------------------------------------------------ #
# Precise timing options
# ------------------------------------------------------------------------------------ #
_C.PREC_TIME = CfgNode()

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 200


# ------------------------------------------------------------------------------------ #
# Memory options
# ------------------------------------------------------------------------------------ #
_C.MEM = CfgNode()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True


# ------------------------------------------------------------------------------------ #
# CUDNN options
# ------------------------------------------------------------------------------------ #
_C.CUDNN = CfgNode()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = True

# ------------------------------------------------------------------------------------ #
# Misc options
# ------------------------------------------------------------------------------------ #

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = "/tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "file"

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port for initializing multi-process groups
_C.HOST = "localhost"
_C.PORT = 10001

# Models weights referred to by URL are downloaded to this local cache
_C.DOWNLOAD_CACHE = "/tmp/pycls-download-cache"

# If we use a determinstic to stablize the search process
_C.DETERMINSTIC = True

# ------------------------------------------------------------------------------------ #
#  keys in DARTS
# ------------------------------------------------------------------------------------ #
_C.DARTS = CfgNode()
_C.DARTS.ALPHA_LR = 3e-4
_C.DARTS.ALPHA_WEIGHT_DECAY = 1e-3
# ------------------------------------------------------------------------------------ #
# Deprecated keys
# ------------------------------------------------------------------------------------ #

_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")
_C.register_deprecated_key("PREC_TIME.ENABLED")


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options.
       How to use: python xx.py --cfg path_to_your_config.cfg test1 0 test2 True
       opts will return a list with ['test1', '0', 'test2', 'True'], yacs will compile to corresponding values
    """
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file",
                        help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)


def assert_and_infer_cfg(cache_urls=True):
    """Checks config values invariants."""
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.SEARCH.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
