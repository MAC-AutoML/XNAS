from xnas.core.trainer import setup_env
from xnas.search_algorithm.darts import *
from xnas.search_space.cell_based import DartsCNN, NASBench201CNN
from xnas.core.config import cfg
from xnas.core.builders import build_space
from xnas.datasets.loader import _construct_loader
import xnas.core.config as config


def main():
    setup_env()
    # loadiong search space
    search_space = build_space()
    # init controller and architecture
    darts_controller = DartsCNNController(search_space)
    architecture = Architect(
        darts_controller, cfg.OPTIM.MOMENTUM, cfg.OPTIM.WEIGHT_DECAY)
    # load dataset
    [train_, val_] = _construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)


if __name__ == "__main__":
    config.load_cfg_fom_args()
    config.assert_and_infer_cfg()
    cfg.freeze()
    main()
