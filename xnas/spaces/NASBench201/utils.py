from collections import namedtuple

from .genos import Structure as CellStructure
from .cnn import TinyNetwork


def dict2config(xdict, logger):
    assert isinstance(xdict, dict), "invalid type : {:}".format(type(xdict))
    Arguments = namedtuple("Configure", " ".join(xdict.keys()))
    content = Arguments(**xdict)
    if hasattr(logger, "log"):
        logger.log("{:}".format(content))
    return content

def get_cell_based_tiny_net(config):
    if hasattr(config, "genotype"):
        genotype = config.genotype
    elif hasattr(config, "arch_str"):
        genotype = CellStructure.str2structure(config.arch_str)
    else:
        raise ValueError(
            "Can not find genotype from this config : {:}".format(config)
        )
    return TinyNetwork(config.C, config.N, genotype, config.num_classes)
