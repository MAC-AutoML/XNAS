import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xnas.search_space.cell_based import *
from xnas.search_space.cell_based import _MixedOp


"""
Pdarts is a search space specific method, which need to change the search space in darts
"""


class _MixOp4Pdarts(_MixedOp):
    '''for pdarts to add dropout for identity'''

    def __init__(self, C_in, C_out, stride, p, basic_op_list=None):
        super(_MixOp4Pdarts, self).__init__(C_in, C_out, stride, basic_op_list)
        self.p = p
        for i, op in enumerate(self._ops):
            if isinstance(op, Identity):
                self._ops[i] = nn.Sequential(op, nn.Dropout(self.p))

    def update_p(self, p):
        for op in self._ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    self.p = p
                    op[1].p = self.p
                    # print(op, op[1].p)


# the search cell in pdarts
class PdartsCell(nn.Module):
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, basic_op_list, dropout_p):
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
        self.basic_op_list = basic_op_list
        self.p = dropout_p

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        pre = 0
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                edg_id = pre+j
                op = _MixOp4Pdarts(C, C, stride, self.p, self.basic_op_list[edg_id])
                self.dag[i].append(op)
        pre += 2+i

    def forward(self, s0, s1, sample):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        w_dag = darts_weight_unpack(sample, self.n_nodes)
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w)
                        for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)
        s_out = torch.cat(states[2:], 1)
        return s_out

    def update_p(self, p):
        for i in range(self.n_nodes):
            for op in self.dag[i]:
                op.update_p(p)


class PdartsCNN(nn.Module):
    def __init__(self, C=16, n_classes=10, n_layers=8, n_nodes=4, basic_op_list=[], p=0.0):
        super().__init__()
        self.p = float(p)
        stem_multiplier = 3
        self.C_in = 3  # 3
        self.C = C  # 16
        self.n_classes = n_classes  # 10
        self.n_layers = n_layers  # 8
        self.n_nodes = n_nodes  # 4
        self.basic_op_list = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
                              'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none',
                              ] if len(basic_op_list) == 0 else basic_op_list
        print("basic_op_list", self.basic_op_list)
        self.len_op = len(self.basic_op_list)
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
            if reduction:
                cell = PdartsCell(n_nodes, C_pp, C_p, C_cur,
                                  reduction_p, reduction, self.basic_op_list[:self.len_op//2], p)
            else:
                cell = PdartsCell(n_nodes, C_pp, C_p, C_cur,
                                  reduction_p, reduction, self.basic_op_list[self.len_op//2:], p)
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

    def forward(self, x, sample):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = sample[self.num_edges:] if cell.reduction else sample[0:self.num_edges]
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def parse_from_numpy(self, alpha, k, basic_op_list=None):
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
        pre = [0, 2, 5, 9]

        # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
        # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
        for i, edges in enumerate(alpha):
            # edges: Tensor(n_edges, n_ops)
            edge_max, primitive_indices = torch.topk(
                torch.tensor(edges[:, :]), 1)  # not ignore 'none'
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
            node_gene = []
            for edge_idx in topk_edge_indices:
                prim_idx = primitive_indices[edge_idx]
                prim = basic_op_list[pre[i]+edge_idx][prim_idx]
                node_gene.append((prim, edge_idx.item()))

            gene.append(node_gene)

        return gene

    def genotype(self, theta):
        Genotype = namedtuple(
            'Genotype', 'normal normal_concat reduce reduce_concat')
        theta_norm = darts_weight_unpack(
            theta[0:self.num_edges], self.n_nodes)
        theta_reduce = darts_weight_unpack(
            theta[self.num_edges:], self.n_nodes)
        gene_normal = self.parse_from_numpy(
            theta_norm, k=2, basic_op_list=self.basic_op_list[:self.len_op//2])
        gene_reduce = self.parse_from_numpy(
            theta_reduce, k=2, basic_op_list=self.basic_op_list[self.len_op//2:])
        concat = range(2, 2+self.n_nodes)  # concat all intermediate nodes
        return Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)

    def update_p(self, p):
        for cell in self.cells:
            # cell.p=self.p
            cell.update_p(p)


def _PdartsCNN():
    from xnas.core.config import cfg
    return PdartsCNN(
        C=cfg.SPACE.CHANNEL+cfg.SEARCH.add_width,
        n_classes=cfg.SPACE.NUM_CLASSES,
        n_layers=cfg.SPACE.LAYERS+cfg.SEARCH.add_layers,
        n_nodes=cfg.SPACE.NODES,
        basic_op_list=cfg.SPACE.BASIC_OP,
        p=float(cfg.SEARCH.dropout_rate))


class PdartsCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, net, criterion, device_ids=None):
        super().__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.net = net
        self.device_ids = device_ids
        self.n_ops = len(self.net.basic_op_list[0])
        self.alpha = nn.Parameter(
            1e-3*torch.randn(self.net.all_edges, self.n_ops))
        self.criterion = criterion

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

    def forward(self, x):
        weights_ = F.softmax(self.alpha, dim=-1)

        if len(self.device_ids) == 1:
            return self.net(x, weights_)
        else:
            raise NotImplementedError
            # multiple GPU support
            # # scatter x
            # xs = nn.parallel.scatter(x, self.device_ids)
            # # broadcast weights
            # wnormal_copies = broadcast_list(weights_normal, self.device_ids)
            # wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

            # # replicate modules
            # replicas = nn.parallel.replicate(self.net, self.device_ids)
            # outputs = nn.parallel.parallel_apply(replicas,
            #                                     list(
            #                                         zip(xs, wnormal_copies, wreduce_copies)),
            #                                     devices=self.device_ids)
            # return nn.parallel.gather(outputs, self.device_ids[0])

    def genotype(self, final=False):
        if final:
            for i in range(self.net.all_edges):
                for j, op in enumerate(self.net.basic_op_list[i]):
                    if op == 'none':
                        self.alpha[i][j] = -1  # 让none操作对应的权值将为最低

        return self.net.genotype(self.alpha.cpu().detach().numpy())

    def get_skip_number(self):
        normal = self.genotype(final=True).normal
        res = 0
        for edg in normal:
            # print(edg)
            if edg[0][0] == 'skip_connect':
                res += 1
            if edg[1][0] == 'skip_connect':
                res += 1
        return res

    def delete_skip(self):
        normal = self.genotype().normal
        pre = 0
        skip_id = []
        skip_edg = []
        pre = [0, 2, 5, 9]
        for i, edg in enumerate(normal):
            for k in range(2):
                if edg[k][0] == 'skip_connect':
                    edg_num = pre[i]+edg[k][1]
                    skip_edg.append(edg_num)
                    for j, op in enumerate(self.net.basic_op_list[edg_num]):
                        if op == 'skip_connect':
                            skip_id.append(j)
                            break

        alpha = self.alpha.cpu().detach().numpy()
        print('basic_op_list', self.net.basic_op_list)
        print('skip_edg', skip_edg)
        print('skip_id', skip_id)
        skip_edg_value = [alpha[pos][skip_id[i]] for i, pos in enumerate(skip_edg)]
        min = np.argmin(skip_edg_value)
        # print('skip_edg[min] skip_id[min]', skip_edg[min], skip_id[min])
        # print('alpha_min')
        for i in range(self.n_ops):
            print(self.alpha[skip_edg[min]][i])
        self.alpha[skip_edg[min]][skip_id[min]] = 0.0

    def get_topk_op(self, k):
        basic_op_list = np.array(self.net.basic_op_list)
        # print(basic_op_list)
        # print(type(basic_op_list))
        # print(type(basic_op_list[0][0]))
        new_basic_op = []
        for i in range(self.net.all_edges):
            _, index = torch.topk(self.alpha[i], k)
            primitive = basic_op_list[i][index.cpu()].tolist()
            new_basic_op.append(primitive)
        return new_basic_op

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def subnet_weights(self):
        res = []
        for k, v in self.named_weights():
            if 'alpha' not in k:
                res.append(v)
        return res

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def alphas_weight(self):
        return self.alpha

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def print_alphas(self, logger):
        logger.info("####### ALPHA #######")
        for alpha in self.alpha:
            logger.info(F.softmax(alpha, dim=-1).cpu().detach().numpy())
        logger.info("#####################")

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def update_p(self, p):
        self.net.p = p
        self.net.update_p(p)

    def remove_kop(self, k):

        pass


class Architect():
    """ Compute gradients of alphas """

    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y)  # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get(
                    'momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y)  # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(
            loss, self.net.alphas())  # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(
            loss, self.net.alphas())  # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
