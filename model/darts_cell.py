""" CNN cell for architecture search """
import torch
import torch.nn as nn
from model import darts_ops as ops
import torch.nn.functional as F
import numpy as np
from utils import utils


# the search cell
class SelectCell(nn.Module):
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
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

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                # change the mixedOp to selectOP
                # op = ops.MixedOp(C, stride)
                op = ops.SelectOp(C, stride)
                self.dag[i].append(op)

    def forward(self, s0, s1, sample):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        w_dag = utils.darts_weight_unpack(sample, self.n_nodes)
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        return s_out


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):

    def __init__(self, n_nodes, C_in, C_out, stride, search_sapce):
        super(NAS201SearchCell, self).__init__()
        self.search_space = search_sapce
        # generate dag
        self.edges = nn.ModuleDict()
        self.in_dim = C_in
        self.out_dim = C_out
        self.n_nodes = n_nodes
        for i in range(1, n_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    self.edges[node_str] = ops.SelectBasicOperation(C_in, C_out, stride, search_space=self.search_space)
                else:
                    self.edges[node_str] = ops.SelectBasicOperation(C_in, C_out, 1, search_space=self.search_space)
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


class InitCell(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        # down sample and not down sample
        self.ops = nn.ModuleList()
        self.ops.append(nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout)
        ))
        self.ops.append(nn.Sequential(
            nn.Conv2d(cin, cout, 3, 2, 1, bias=False),
            nn.BatchNorm2d(cout)
        ))

    def forward(self, x, sample):
        op = self.ops[int(np.argmax(sample))]
        return F.relu(op(x))


class NormCell(nn.Module):
    def __init__(self, cin, cout, n_blocks=5):
        super().__init__()
        self.ops = nn.ModuleList()
        self.block = ops.BasicBlock
        if cin == cout:
            self.ops.append(ops.Identity())
            for i in range(n_blocks):
                sub_ops = []
                for j in range(2**i):
                    sub_ops.append(self.block(cin, cout))
                self.ops.append(nn.Sequential(*sub_ops))
        else:
            self.ops.append(self.block(cin, cout))
            self.ops.append(self.block(cin, cout, stride=2))

    def forward(self, x, sample):
        op = self.ops[int(np.argmax(sample))]
        return op(x)

