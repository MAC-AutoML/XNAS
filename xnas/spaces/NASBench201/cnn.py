import random
import numpy as np
import torch.nn as nn
from copy import deepcopy

from .ops import (
    OPS, 
    ResNetBasicblock, 
    get_op_index,
    NON_PARAMETER_OP,
    PARAMETER_OP
)


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
    ):
        super(NAS201SearchCell, self).__init__()
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, self.max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    xlists = [
                        OPS[op_name](C_in, C_out, stride, affine, track_running_stats)
                        for op_name in op_names
                    ]
                else:
                    xlists = [
                        OPS[op_name](C_in, C_out, 1, affine, track_running_stats)
                        for op_name in op_names
                    ]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    sum(
                        layer(nodes[j]) * w
                        for layer, w in zip(self.edges[node_str], weights)
                    )
                )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # GDAS
    def forward_gdas(self, inputs, hardwts, index):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = hardwts[self.edge2index[node_str]]
                argmaxs = index[self.edge2index[node_str]].item()
                weigsum = sum(
                    weights[_ie] * edge(nodes[j]) if _ie == argmaxs else weights[_ie]
                    for _ie, edge in enumerate(self.edges[node_str])
                )
                inter_nodes.append(weigsum)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # joint
    def forward_joint(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                # aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
                aggregation = sum(
                    layer(nodes[j]) * w
                    for layer, w in zip(self.edges[node_str], weights)
                )
                inter_nodes.append(aggregation)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # uniform random sampling per iteration, SETN
    def forward_urs(self, inputs):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            while True:  # to avoid select zero for all ops
                sops, has_non_zero = [], False
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    candidates = self.edges[node_str]
                    select_op = random.choice(candidates)
                    sops.append(select_op)
                    if not hasattr(select_op, "is_zero") or select_op.is_zero is False:
                        has_non_zero = True
                if has_non_zero:
                    break
            inter_nodes = []
            for j, select_op in enumerate(sops):
                inter_nodes.append(select_op(nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # select the argmax
    def forward_select(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    self.edges[node_str][weights.argmax().item()](nodes[j])
                )
                # inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # forward with a specific structure
    def forward_dynamic(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i - 1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = "{:}<-{:}".format(i, j)
                op_index = self.op_names.index(op_name)
                inter_nodes.append(self.edges[node_str][op_index](nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

class SPOS_NAS201SearchCell(NAS201SearchCell):
    # select the argmax
    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    self.edges[node_str][weights](nodes[j])
                )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

class NASBench201CNN(nn.Module):
    def __init__(self, C=16, N=5, max_nodes=4, num_classes=10, basic_op_list=[]):
        super(NASBench201CNN, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.basic_op_list = ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'none'] \
            if len(basic_op_list) == 0 else basic_op_list
        self.non_op_idx = get_op_index(self.basic_op_list, NON_PARAMETER_OP)
        self.para_op_idx = get_op_index(self.basic_op_list, PARAMETER_OP)
        self.none_idx = 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))

        layer_channels = [C] * N + [C * 2] + \
            [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + \
            [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201SearchCell(
                    C_prev, C_curr, 1, self.max_nodes, self.basic_op_list)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(
                        num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.num_edges = num_edge
        self.all_edges = self.num_edges
        self.num_ops = len(self.basic_op_list)
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def genotype(self, theta):
        genotypes = ''
        for i in range(1, self.max_nodes):
            sub_geno = '|'
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = theta[self.edge2index[node_str]]
                op_name = self.basic_op_list[np.argmax(weights)]
                sub_geno += '{0}~{1}|'.format(op_name, str(j))
            if i == 1:
                genotypes += sub_geno
            else:
                genotypes += '+' + sub_geno
        return genotypes

    def weights(self):
        return self.parameters()

    def forward(self, inputs, weight):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, ResNetBasicblock):
                feature = cell(feature)
            else:
                feature = cell(feature, weight)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

class SPOS_nb201_CNN(NASBench201CNN):
    def __init__(self, C=16, N=5, max_nodes=4, num_classes=10, basic_op_list=[]):
        nn.Module.__init__(self)
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.basic_op_list = ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'none'] \
            if len(basic_op_list) == 0 else basic_op_list
        self.non_op_idx = get_op_index(self.basic_op_list, NON_PARAMETER_OP)
        self.para_op_idx = get_op_index(self.basic_op_list, PARAMETER_OP)
        self.none_idx = 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))

        layer_channels = [C] * N + [C * 2] + \
            [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + \
            [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SPOS_NAS201SearchCell(
                    C_prev, C_curr, 1, self.max_nodes, self.basic_op_list)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(
                        num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.num_edges = num_edge
        self.all_edges = self.num_edges
        self.num_ops = len(self.basic_op_list)
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.spos_all_edge_num = 3 * self._layerN * self.all_edges

    def forward(self, inputs, weight):
        feature = self.stem(inputs)
        _it = 0
        for i, cell in enumerate(self.cells):
            if isinstance(cell, ResNetBasicblock):
                feature = cell(feature)
            else:
                feature = cell(feature, weight[_it:_it+self.all_edges])
                _it += self.all_edges
        assert _it == self.spos_all_edge_num
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

# Infer cell for NAS-Bench-201
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

    # def feature_extractor(self, inputs):
    #     features = []
    #     feature = self.stem(inputs)
    #     features.append(feature)

    #     for i, cell in enumerate(self.cells):
    #         feature = cell(feature)
    #         features.append(feature)
    #     out = self.lastact(feature)
    #     features.append(out)
    #     return features

    def forward(self, inputs):
        feature = self.stem(inputs)

        for i, cell in enumerate(self.cells):
            feature = cell(feature)
        feature = self.lastact(feature)

        out = self.global_pooling(feature)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits
    
    def feature_extractor(self, inputs):
        """Used by RMINAS, extract features with logits together."""
        features = []
        feature = self.stem(inputs)

        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            if i == 4:
                tensor1 = feature
            elif i == 10:
                tensor2 = feature
        feature = self.lastact(feature)
        tensor3 = feature
        features = [tensor1, tensor2, tensor3]
        return features


    def forward_with_features(self, inputs):
        """Used by RMINAS, extract features with logits together."""
        features = []
        feature = self.stem(inputs)

        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            if i == 4:
                tensor1 = feature
            elif i == 10:
                tensor2 = feature
        feature = self.lastact(feature)
        tensor3 = feature
        features = [tensor1, tensor2, tensor3]

        out = self.global_pooling(feature)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return features, logits


# build API

def _NASBench201():
    from xnas.core.config import cfg
    return NASBench201CNN(
        C=cfg.SPACE.CHANNELS,
        N=cfg.SPACE.LAYERS,
        max_nodes=cfg.SPACE.NODES,
        num_classes=cfg.LOADER.NUM_CLASSES,
        basic_op_list=cfg.SPACE.BASIC_OP
    )

def _SPOS_nb201_CNN():
    from xnas.core.config import cfg
    return SPOS_nb201_CNN(
        C=cfg.SPACE.CHANNELS,
        N=cfg.SPACE.LAYERS,
        max_nodes=cfg.SPACE.NODES,
        num_classes=cfg.LOADER.NUM_CLASSES,
        basic_op_list=cfg.SPACE.BASIC_OP
    )
    
def _infer_NASBench201():
    from xnas.core.config import cfg
    return TinyNetwork(
        C=cfg.TRAIN.CHANNELS,
        N=cfg.TRAIN.LAYERS,
        genotype=cfg.TRAIN.GENOTYPE,
        num_classes=cfg.LOADER.NUM_CLASSES,
    )
