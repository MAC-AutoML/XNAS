import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from xnas.search_space.cell_based import *

'''
Darts: highly copyed from https://github.com/khanrc/pt.darts
'''


class PCDartsCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, net, criterion, device_ids=None):
        super().__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.net = net
        self.device_ids = device_ids
        self.n_ops = len(self.net.basic_op_list)
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

    def genotype(self):
        return self.net.genotype(self.alpha.cpu().detach().numpy())

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

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


class PcDartsCNN(DartsCNN):
    def __init__(self, C=16, n_classes=10, n_layers=8, n_nodes=4, basic_op_list=[]):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = PcDartsCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, x, sample):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def genotype(self):
        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]

                # edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps - 1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def arch_parameters(self):
        return self._arch_parameters

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

class PcDartsCell(DartsCell):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
