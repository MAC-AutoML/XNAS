from xnas.search_space.cellbased_basic_ops import *


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG


class NASBench201Cell(nn.Module):

    def __init__(self, n_nodes, C_in, C_out, stride, basic_op_list):
        super(NASBench201Cell, self).__init__()
        self.basic_op_list = basic_op_list
        # generate dag
        self.edges = nn.ModuleDict()
        self.in_dim = C_in
        self.out_dim = C_out
        self.n_nodes = n_nodes
        for i in range(1, n_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    self.edges[node_str] = MixedOp(
                        C_in, C_out, stride, basic_op_list)
                else:
                    self.edges[node_str] = MixedOp(
                        C_in, C_out, 1, basic_op_list)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def forward(self, inputs, sample):
        nodes = [inputs]
        for i in range(1, self.n_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = sample[self.edge2index[node_str]]
                inter_nodes.append(self.edges[node_str](nodes[j], weights))
            nodes.append(sum(inter_nodes))
        return nodes[-1]


class NASBench201CNN(nn.Module):
    # def __init__(self, C, N, max_nodes, num_classes, search_space, affine=False, track_running_stats=True):
    def __init__(self, C=16, N=5, max_nodes=4, num_classes=10, basic_op_list=[]):
        super(NASBench201CNN, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.basic_op_list = ['none', 'skip_connect', 'nor_conv_1x1',
                              'nor_conv_3x3', 'avg_pool_3x3'] if len(basic_op_list) == 0 else basic_op_list
        self.non_op_idx = get_op_index(self.basic_op_list, NON_PARAMETER_OP)
        self.para_op_idx = get_op_index(self.basic_op_list, PARAMETER_OP)
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
                cell = NASBench201Cell(
                    max_nodes, C_prev, C_curr, 1, self.basic_op_list)
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


# build API

def _NASBench201():
    from xnas.core.config import cfg
    return NASBench201CNN(C=cfg.SPACE.CHANNEL,
                          N=cfg.SPACE.LAYERS,
                          max_nodes=cfg.SPACE.NODES,
                          num_classes=cfg.SEARCH.NUM_CLASSES,
                          basic_op_list=cfg.SPACE.BASIC_OP)