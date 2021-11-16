import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import copy


'''
Darts: highly copyed from https://github.com/khanrc/pt.darts
'''


class DartsCNNController(nn.Module):
    """SearchCNN Controller"""

    def __init__(self, net, criterion, device_ids=None):
        super().__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.net = net
        self.device_ids = device_ids
        self.n_ops = self.net.num_ops
        self.alpha = nn.Parameter(
            1e-3*torch.randn(self.net.all_edges, self.n_ops))
        self.criterion = criterion

        # Setup alphas list
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

    def genotype(self):
        """return genotype of DARTS CNN"""
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

        # virtual step (update gradient)
        # operations below do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer,
            # So original network weight have to be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get(
                    'momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, unrolled=True):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        if unrolled:
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
                    alpha.grad = da - xi * h
            
        else:
            loss = self.net.loss(val_X, val_y)  # L_trn(w)
            loss.loss.backward()


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
            loss, self.net.alphas())    # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(
            loss, self.net.alphas())    # dalpha { L_trn(w-) }
        
        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        
        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian