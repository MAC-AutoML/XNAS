##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

from collections import namedtuple
import torch.nn as nn
from xnas.search_space.RMINAS.NB201.ops import OPS, ResNetBasicblock

from copy import deepcopy

from xnas.search_space.RMINAS.NB201.geno import Structure as CellStructure


@staticmethod
def str2lists(arch_str):
    """
    This function shows how to read the string-based architecture encoding.
      It is the same as the `str2structure` func in `AutoDL-Projects/lib/models/cell_searchs/genotypes.py`
    :param
      arch_str: the input is a string indicates the architecture topology, such as
                    |nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|
    :return: a list of tuple, contains multiple (op, input_node_index) pairs.
    :usage
      arch = api.str2lists( '|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|' )
      print ('there are {:} nodes in this arch'.format(len(arch)+1)) # arch is a list
      for i, node in enumerate(arch):
        print('the {:}-th node is the sum of these {:} nodes with op: {:}'.format(i+1, len(node), node))
    """
    node_strs = arch_str.split('+')
    genotypes = []
    for i, node_str in enumerate(node_strs):
        inputs = list(filter(lambda x: x != '', node_str.split('|')))
        for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
        inputs = ( xi.split('~') for xi in inputs )
        input_infos = tuple( (op, int(IDX)) for (op, IDX) in inputs)
        genotypes.append( input_infos )
    return genotypes

def dict2config(xdict, logger):
    assert isinstance(xdict, dict), "invalid type : {:}".format(type(xdict))
    Arguments = namedtuple("Configure", " ".join(xdict.keys()))
    content = Arguments(**xdict)
    if hasattr(logger, "log"):
        logger.log("{:}".format(content))
    return content

def config2dict(content):
    return content._asdict()

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


# Cell for NAS-Bench-201
class InferCell(nn.Module):
    def __init__(
        self, genotype, C_in, C_out, stride, affine=True, track_running_stats=True
    ):
        super(InferCell, self).__init__()

        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                if op_in == 0:
                    layer = OPS[op_name](
                        C_in, C_out, stride, affine, track_running_stats
                    )
                else:
                    layer = OPS[op_name](C_out, C_out, 1, affine, track_running_stats)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def extra_repr(self):
        string = "info :: nodes={nodes}, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        laystr = []
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = [
                "I{:}-L{:}".format(_ii, _il)
                for _il, _ii in zip(node_layers, node_innods)
            ]
            x = "{:}<-({:})".format(i + 1, ",".join(y))
            laystr.append(x)
        return (
            string
            + ", [{:}]".format(" | ".join(laystr))
            + ", {:}".format(self.genotype.tostr())
        )

    def forward(self, inputs):
        nodes = [inputs]
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            node_feature = sum(
                self.layers[_il](nodes[_ii])
                for _il, _ii in zip(node_layers, node_innods)
            )
            nodes.append(node_feature)
        return nodes[-1]


# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(nn.Module):
    def __init__(self, C, N, genotype, num_classes):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N
        # self._datasize = datasize
        # self._feature_res = feature_res

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )
    
    def feature_extractor(self, inputs):
        features = []
        feature = self.stem(inputs)
        features.append(feature)
        
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            features.append(feature)
        out = self.lastact(feature)
        features.append(out)
        return features

    def forward(self, inputs):
        features = []
        feature = self.stem(inputs)

        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            if i == 4:
                tensor1 = feature
            elif i == 10:
                tensor2 = feature
            # if i in [4,10]:
                # features.append(feature)
        feature = self.lastact(feature)
        tensor3 = feature

        features = [tensor1, tensor2, tensor3]
        # features.append(feature)

        out = self.global_pooling(feature)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return features, logits