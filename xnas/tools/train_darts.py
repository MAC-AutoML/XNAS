from xnas.core.trainer import setup_env
from xnas.search_algorithm.darts import *
from xnas.search_space.cell_based import DartsCNN, NASBench201CNN
from xnas.core.config import cfg
import xnas.core.config as config


def main():
    setup_env()
    # loadiong search space
    

if __name__ == "__main__":
    config.load_cfg_fom_args()
    config.assert_and_infer_cfg()
    cfg.freeze()
    main()
