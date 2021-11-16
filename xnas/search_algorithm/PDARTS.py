import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Pdarts is a search space specific method, which need to change the search space in darts
"""


class PDartsCNNController(nn.Module):
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
