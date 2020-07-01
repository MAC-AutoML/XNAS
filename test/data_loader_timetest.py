#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compute model and loader timings."""
import os
import xnas.core.benchmark as benchmark
import xnas.core.config as config
import xnas.core.logging as logging
import xnas.datasets.loader as loader
import xnas.core.distributed as dist
from xnas.core.config import cfg


logger = logging.get_logger(__name__)


def test_full_time():
    # Save the config
    config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # config.assert_and_infer_cfg()
    test_loader = loader.construct_test_loader()
    avg_time = benchmark.compute_full_loader(test_loader, epoch=3)
    for i, _time in enumerate(avg_time):
        logger.info("The {}'s epoch average time is: {}".format(i, _time))


def main():
    config.load_cfg_fom_args("Compute model and loader timings.")
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dist.multi_proc_run(num_proc=1, fun=test_full_time)


if __name__ == "__main__":
    main()
