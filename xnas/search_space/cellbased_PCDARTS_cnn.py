import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from xnas.search_space.cellbased_basic_ops import *
from torch.autograd import Variable


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class PcMixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, basic_op_list=None):
        super().__init__()

        self.k = 4
        self.mp = nn.MaxPool2d(2, 2)
        self._ops = nn.ModuleList()
        assert basic_op_list is not None, "the basic op list cannot be none!"
        basic_primitives = basic_op_list

        for primitive in basic_primitives:
            op = OPS_[primitive](C_in//self.k, C_out//self.k, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2//self.k, :, :]
        xtemp2 = x[:,  dim_2//self.k:, :, :]
        assert len(self._ops) == len(weights)
        '''
        temp1 = 0
        for i, value in enumerate(weights):
            if value == 1:
                temp1 += self._ops[i](xtemp)
            if 0 < value < 1:
                temp1 += value * self._ops[i](xtemp)'''
        _x = []
        for i, value in enumerate(weights):
            if value == 1:
                _x.append(self._ops[i](xtemp))
            if 0 < value < 1:
                _x.append(value * self._ops[i](xtemp))

        # reduction cell needs pooling before concat
        part_x = sum(_x)
        if part_x.shape[2] == x.shape[2]:
            ans = torch.cat([part_x, xtemp2], dim=1)
        else:
            ans = torch.cat([part_x, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, self.k)
        # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans


# the search cell in darts


class PcDartsCell(nn.Module):
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, basic_op_list, multiplier):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self._multiplier = multiplier
        self.basic_op_list = basic_op_list

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = PcMixedOp(C, C, stride, self.basic_op_list)
                self.dag[i].append(op)

    def forward(self, s0, s1, sample, sample2):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        w_dag = darts_weight_unpack(sample, self.n_nodes)
        w_w_dag = darts_weight_unpack(sample2, self.n_nodes)

        for edges, w_list, w_w_list in zip(self.dag, w_dag, w_w_dag):
            s_cur = sum(ww * edges[i](s, w)
                for i, (s, w, ww) in enumerate(zip(states, w_list, w_w_list)))
            states.append(s_cur)
        s_out = torch.cat(states[-self._multiplier:], 1)
        return s_out


# PcDartsCNN


class PcDartsCNN(nn.Module):

    def __init__(self, C=16, n_classes=10, n_layers=8, n_nodes=4, basic_op_list=[], multiplier=4):
        super().__init__()
        stem_multiplier = 3
        self._multiplier = multiplier
        self.C_in = 3  # 3
        self.C = C  # 16
        self.n_classes = n_classes  # 10
        self.n_layers = n_layers  # 8
        self.n_nodes = n_nodes  # 4
        self.basic_op_list = ['none','max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
                              'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5' ] if len(basic_op_list) == 0 else basic_op_list
        C_cur = stem_multiplier * C  # 3 * 16 = 48
        self.stem = nn.Sequential(
            nn.Conv2d(self.C_in, C_cur, 3, 1, 1, bias=False),
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
            cell = PcDartsCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction, self.basic_op_list, multiplier)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
        # number of edges per cell
        self.num_edges = sum(list(range(2, self.n_nodes + 2)))
        # whole edges
        self.all_edges = 2 * self.num_edges

    def forward(self, x, sample, sample2):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                alphas_reduce = sample[self.num_edges:]
                betas_reduce = sample2[self.num_edges:]
                weights = F.softmax(alphas_reduce, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(betas_reduce[0:2], dim=-1)
                for i in range(self.n_nodes - 1):
                    end = start + n
                    tw2 = F.softmax(betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                alphas_normal = sample[0:self.num_edges]
                betas_normal = sample2[0:self.num_edges]
                weights = F.softmax(alphas_normal, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(betas_normal[0:2], dim=-1)
                for i in range(self.n_nodes - 1):
                    end = start + n
                    tw2 = F.softmax(betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def genotype(self, theta, theta2):

        Genotype = namedtuple(
            'Genotype', 'normal normal_concat reduce reduce_concat')
        a_norm = theta[0:self.num_edges]
        a_reduce = theta[self.num_edges:]
        b_norm = theta2[0:self.num_edges]
        b_reduce = theta2[self.num_edges:]
        weightn = F.softmax(a_norm, dim=-1)
        weightr = F.softmax(a_reduce, dim=-1)
        n = 3
        start = 2
        weightsn2 = F.softmax(b_norm[0:2], dim=-1)
        weightsr2 = F.softmax(b_reduce[0:2], dim=-1)

        for i in range(self.n_nodes - 1):
            end = start + n
            tn2 = F.softmax(b_norm[start:end], dim=-1)
            tw2 = F.softmax(b_reduce[start:end], dim=-1)
            start = end
            n += 1
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)

        theta_norm = darts_weight_unpack(weightn, self.n_nodes)
        theta_reduce = darts_weight_unpack(weightr, self.n_nodes)
        theta2_norm = darts_weight_unpack(weightsn2, self.n_nodes)
        theta2_reduce = darts_weight_unpack(weightsr2, self.n_nodes)

        for t, etheta in enumerate(theta_norm):
            for tt, eetheta in enumerate(etheta):
                theta_norm[t][tt] *= theta2_norm[t][tt]
        for t, etheta in enumerate(theta_reduce):
            for tt, eetheta in enumerate(etheta):
                theta_reduce[t][tt] *= theta2_reduce[t][tt]

        gene_normal = pc_parse_from_numpy(
            theta_norm, k=2, basic_op_list=self.basic_op_list)
        gene_reduce = pc_parse_from_numpy(
            theta_reduce, k=2, basic_op_list=self.basic_op_list)
        concat = range(2 + self.n_nodes - self._multiplier, 2 + self.n_nodes)  # concat all intermediate nodes
        return Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)


def pc_parse_from_numpy(alpha, k, basic_op_list=None):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert basic_op_list[0] == 'none'  # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(
            torch.tensor(edges[:, 1:]), 1)  # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = basic_op_list[prim_idx+1]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene

def _PcdartsCNN():
    from xnas.core.config import cfg
    return PcDartsCNN(
        C=cfg.SPACE.CHANNEL,
        n_classes=cfg.SEARCH.NUM_CLASSES,
        n_layers=cfg.SPACE.LAYERS,
        n_nodes=cfg.SPACE.NODES,
        basic_op_list=cfg.SPACE.BASIC_OP)
