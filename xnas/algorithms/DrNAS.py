import torch
import numpy as np
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, net, cfg):
        self.network_momentum = cfg.OPTIM.MOMENTUM
        self.network_weight_decay = cfg.OPTIM.WEIGHT_DECAY
        self.net = net
        if cfg.DRNAS.REG_TYPE == "l2":
            weight_decay = cfg.DRNAS.REG_SCALE
        elif cfg.DRNAS.REG_TYPE == "kl":
            weight_decay = 0
        self.optimizer = torch.optim.Adam(
            self.net.arch_parameters(),
            lr=cfg.DARTS.ALPHA_LR,
            betas=(0.5, 0.999),
            weight_decay=weight_decay,
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.net._loss(input, target)
        theta = _concat(self.net.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]["momentum_buffer"]
                for v in self.net.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = (
            _concat(torch.autograd.grad(loss, self.net.parameters())).data
            + self.network_weight_decay * theta
        )
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta)
        ).to(self.device)
        return unrolled_model

    def unrolled_backward(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
        unrolled,
    ):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train,
                target_train,
                input_valid,
                target_valid,
                eta,
                network_optimizer,
            )
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    # def pruning(self, masks):
    #   for i, p in enumerate(self.optimizer.param_groups[0]['params']):
    #     if masks[i] is None:
    #       continue
    #     state = self.optimizer.state[p]
    #     mask = masks[i]
    #     state['exp_avg'][~mask] = 0.0
    #     state['exp_avg_sq'][~mask] = 0.0

    def _backward_step(self, input_valid, target_valid):
        loss = self.net._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
    ):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer
        )
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.net.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.net.new()
        model_dict = self.net.state_dict()

        params, offset = {}, 0
        for k, v in self.net.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset : offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.net.parameters(), vector):
            p.data.add_(R, v)
        loss = self.net._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.net.arch_parameters())

        for p, v in zip(self.net.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.net._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.net.arch_parameters())

        for p, v in zip(self.net.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
