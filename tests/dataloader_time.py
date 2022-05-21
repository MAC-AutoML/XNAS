# Test the time cost of constructing dataloader.

# import xnas.core.logging as logging
# import torch
# from xnas.core.config import cfg
# from xnas.core.timer import Timer

# import os
# import xnas.core.benchmark as benchmark
# import xnas.core.config as config
# import xnas.core.logging as logging
# import xnas.datasets.loader as loader
# import xnas.core.distributed as dist
# from xnas.core.config import cfg


# logger = logging.get_logger(__name__)





# def compute_full_loader(data_loader, epoch=1):
#     """Computes full loader time."""
#     timer = Timer()
#     epoch_avg = []
#     data_loader_len = len(data_loader)
#     for j in range(epoch):
#         timer.tic()
#         for i, (inputs, labels) in enumerate(data_loader):
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             timer.toc()
#             logger.info("Epoch {}/{}, Iter {}/{}: Dataloader time is: {}".format(j + 1, epoch, i+1, data_loader_len, timer.diff))
#             timer.tic()
#         epoch_avg.append(timer.average_time)
#         timer.reset()
#     return epoch_avg



# def test_full_time():
#     config.dump_cfg()
#     logging.setup_logging()
#     logger.info("Config:\n{}".format(cfg))
#     logger.info(logging.dump_log_data(cfg, "cfg"))

#     [train_loader, test_loader] = loader.construct_loader(
#         cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE, cfg.SEARCH.DATAPATH)

#     avg_time = benchmark.compute_full_loader(test_loader, epoch=3)

#     for i, _time in enumerate(avg_time):
#         logger.info("The {}'s epoch average time is: {}".format(i, _time))

# if __name__ == "__main__":
#     config.load_cfg_fom_args("Compute model and loader timings.")
#     os.makedirs(cfg.OUT_DIR, exist_ok=True)
#     dist.multi_proc_run(num_proc=1, fun=test_full_time)
