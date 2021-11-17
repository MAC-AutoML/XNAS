import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable


class PCDartsCNNController(nn.Module):
    """SearchCNN controller"""

    def __init__(self, net, criterion, device_ids=None):
        super().__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.net = net
        self.device_ids = device_ids
        self.n_ops = len(self.net.basic_op_list)
        an = 1e-3 * torch.randn(self.net.all_edges//2, self.n_ops)
        ar = 1e-3 * torch.randn(self.net.all_edges//2, self.n_ops)
        al = torch.cat([an,ar],0)
        self.alpha = nn.Parameter(al)
        bn = 1e-3 * torch.randn(self.net.all_edges//2)
        br = 1e-3 * torch.randn(self.net.all_edges//2)
        bl = torch.cat([bn,br],0)
        self.beta = nn.Parameter(bl)

        self.criterion = criterion

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        # print(self._alphas)
        # setup beta list
        self._betas = []
        for n, p in self.named_parameters():
            if 'beta' in n:
                self._betas.append((n, p))
        # (self._betas)

    def forward(self, x):
        # weights_1 = F.softmax(self.alpha, dim=-1)
        # weights_2 = F.softmax(self.beta, dim=-1)

        if len(self.device_ids) == 1:
            return self.net(x, self.alpha, self.beta)
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
        return self.net.genotype(self.alpha, self.beta)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p
        for n, p in self._betas:
            yield p

    # def betas(self):
    #    for n, p in self._betas:
    #        yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def named_betas(self):
        for n, p in self._betas:
            yield n, p

    def print_alphas(self, logger):
        logger.info("####### ALPHA #######")
        for alpha in self.alpha:
            logger.info(F.softmax(alpha, dim=-1).cpu().detach().numpy())
        logger.info("####### BETA #######")
        n_nodes = 4
        num_edges = 14
        b_norm = self.beta[0:num_edges]
        b_reduce = self.beta[num_edges:]
        n = 3
        start = 2
        weightsn2 = F.softmax(b_norm[0:2], dim=-1)
        weightsr2 = F.softmax(b_reduce[0:2], dim=-1)

        for i in range(n_nodes - 1):
            end = start + n
            tn2 = F.softmax(b_norm[start:end], dim=-1)
            tw2 = F.softmax(b_reduce[start:end], dim=-1)
            start = end
            n += 1
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
        for beta in weightsn2:
            logger.info(beta.cpu().detach().numpy())
        for beta in weightsr2:
            logger.info(beta.cpu().detach().numpy())
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
            # for b, vb in zip(self.net.betas(), self.v_net.betas()):
            #    vb.copy_(b)

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
        v_weights = tuple(self.v_net.weights())
        # alpha
        v_alphas = tuple(self.v_net.alphas())
        # v_betas = tuple(self.v_net.betas())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights, retain_graph=True)
        dalpha = v_grads[:len(v_alphas)]
        # dbeta = v_grads[len(v_alphas):len(v_alphas)+len(v_betas)]
        dw = v_grads[len(v_alphas):]

        hessiana = self.compute_hessian(dw, trn_X, trn_y)
        # hessianb = self.compute_hessian(dw, trn_X, trn_y, 1)
        # update final gradient = dalpha - xi*hessian
        # with torch.no_grad():

        # beta

        # update final gradient = dbeta - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessiana):
                alpha.grad = da - xi * h
            # for beta, db, h in zip(self.net.betas(), dbeta, hessianb):
            #    beta.grad = db - xi * h

    def compute_hessian(self, dw, trn_X, trn_y, bool_beta=0):
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
        if bool_beta == 0:
            dalpha_pos = torch.autograd.grad(
                loss, self.net.alphas())  # dalpha { L_trn(w+) }
        else:
            dalpha_pos = torch.autograd.grad(
                loss, self.net.betas())  # dbeta { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        if bool_beta == 0:
            dalpha_neg = torch.autograd.grad(
                loss, self.net.alphas())  # dalpha { L_trn(w-) }
        else:
            dalpha_neg = torch.autograd.grad(
                loss, self.net.betas())  # dbeta { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
