from model.darts_cell import *
import utils.genotypes as gt
import numpy as np
from model.base_module import MyModule, MyNetwork
from utils.genotypes import NAS_BENCH_201


# search cnn
class SelectSearchCNN(MyNetwork):

    def __init__(self, C_in=3, C=16, n_classes=10, n_layers=8, n_nodes=4, net_ceri=None):
        super().__init__()
        stem_multiplier = 3
        self.criterion = net_ceri
        self.C_in = C_in  # 3
        self.C = C  # 16
        self.n_classes = n_classes  # 10
        self.n_layers = n_layers  # 8
        self.n_nodes = n_nodes  # 4
        C_cur = stem_multiplier * C  # 3 * 16 = 48
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C
        # 48   48   16
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = SelectCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
        # number of edges per cell
        self.num_edges = sum(list(range(2, self.n_nodes + 2)))

    def forward(self, x, sample):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = sample[self.num_edges:] if cell.reduction else sample[0:self.num_edges]
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def genotype(self, theta):
        theta_norm = utils.darts_weight_unpack(theta[0:self.num_edges], self.n_nodes)
        theta_reduce = utils.darts_weight_unpack(theta[self.num_edges:], self.n_nodes)
        gene_normal = gt.parse_numpy(theta_norm, k=2)
        gene_reduce = gt.parse_numpy(theta_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)


class NASBench201CNN(MyNetwork):
    # def __init__(self, C, N, max_nodes, num_classes, search_space, affine=False, track_running_stats=True):
    def __init__(self, C, N, max_nodes, num_classes, search_space):
        super(NASBench201CNN, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ops.ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201SearchCell(max_nodes, C_prev, C_curr, 1, search_space)
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
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def genotype(self, theta):
        genotypes = ''
        for i in range(1, self.max_nodes):
            sub_geno = '|'
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = theta[self.edge2index[node_str]]
                op_name = NAS_BENCH_201[np.argmax(weights)]
                sub_geno += '{0}~{1}|'.format(op_name, str(j))
            if i == 1:
                genotypes += sub_geno
            else:
                genotypes += '+' + sub_geno
        return genotypes

    def forward(self, inputs, weight):

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, ops.ResNetBasicblock):
                feature = cell(feature)
            else:
                feature = cell(feature, weight)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits