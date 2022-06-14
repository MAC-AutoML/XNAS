import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast

import xnas.spaces.DARTS.genos as gt


__all__ = ['DropNAS_CNNController']


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i + len(l)] for i in range(0, len(l_copies), len(l))]
    return l_copies


class DropNAS_CNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, net, criterion, n_nodes=4, device_ids=None):
        super().__init__()
        self.net = net
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        # self.net = DropNASCNN(C_in=C_in, C=C, n_classes=n_classes, n_layers=n_layers)
        # self.net = DropNASCNN(input_size, C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, drop_rate):
        self.masks_normal = self.generate_masks(self.alpha_normal, drop_rate)
        self.masks_reduce = self.generate_masks(self.alpha_reduce, drop_rate)

        weights_normal, self.ratios_normal = self.generate_weights(self.alpha_normal, self.masks_normal)
        weights_reduce, self.ratios_reduce = self.generate_weights(self.alpha_reduce, self.masks_reduce)

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce, self.masks_normal, self.masks_reduce)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)
        mnormal_copies = broadcast_list(self.masks_normal, self.device_ids)
        mreduce_copies = broadcast_list(self.masks_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies, mnormal_copies,
                                                      mreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def print_alphas(self, logger):
        logger.info("####### ALPHA #######")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

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

    def generate_masks(self, weights, drop_rate):

        def generate_mask(drop_prob, length):
            # generate a list of boolean as the mask for each group, and resample if all zero
            while True:
                mask = [random.random() > drop_prob for _ in range(length)]
                if sum(mask) > 0:
                    return mask

        masks = []  # List[List[List[bool]]]], mask for the whole cell
        for ws in weights:
            s1, _ = ws.shape
            drop_prob_para = (drop_rate) ** (1 / 4)
            drop_prob_nopara = (drop_rate) ** (1 / 4)
            mask = torch.Tensor([generate_mask(drop_prob_para, length=4) + generate_mask(drop_prob_nopara, length=4)
                                 for _ in range(s1)]).to(self.device)  # mask for each node
            # the length of the second mask list is set to 2 in the `1-skip` search space
            masks.append(mask)

        return masks

    def generate_weights(self, alphas, masks):
        weights = []
        ratios = []

        for alpha, mask in zip(alphas, masks):
            # for each cell
            weight = torch.empty_like(alpha)
            ratio = []  # ratio means the total probability of the kept operations on each edge

            for i in range(alpha.size(0)):
                # for each edge
                denominator = sum([torch.exp(a) for a, m in zip(alpha[i], mask[i]) if m])
                weight[i] = torch.exp(alpha[i]) / denominator
                ratio.append(denominator.item() / torch.sum(torch.exp(alpha[i])).item())

            weights.append(weight)
            ratios.append(ratio)

        return weights, ratios

    def adjust_alphas(self):

        def adjust(alphas, masks, ratios):
            for alpha, mask, ratio in zip(alphas, masks, ratios):  # for each cell
                for i in range(alpha.size(0)):  # for each edge
                    if sum(mask[i]) < len(gt.PRIMITIVES):  # if there's any dropped operation
                        # The following part works in the same way as introduced in the paper, but more redundant
                        updated_sum = sum([torch.exp(a) for a, m in zip(alpha[i], mask[i]) if m])
                        remain_sum = sum([torch.exp(a) for a, m in zip(alpha[i], mask[i]) if not m])
                        k = math.log(ratio[i] / (1 - ratio[i]) * remain_sum / updated_sum)
                        for a, m in zip(alpha[i], mask[i]):
                            if m:
                                a.data.add_(k)

        adjust(self.alpha_normal, self.masks_normal, self.ratios_normal)
        adjust(self.alpha_reduce, self.masks_reduce, self.ratios_reduce)

    def weight_decay_loss(self, w_decay_rate):
        # conduct partial-decay for weights
        loss = 0
        for w in self.weights():
            if w.grad is not None and not w.grad.view(-1)[0].item() == 0.:
                loss += torch.pow(w.norm(2), 2)

        return loss * w_decay_rate

    def alpha_decay_loss(self, alpha_decay_rate):
        # conduct partial-decay for alphas
        loss = 0
        for a in self.alphas():
            decay_idx = (a.grad.abs() > 1e-7).float()
            decay_alpha = a.mul(decay_idx)
            loss += torch.sum(torch.pow(decay_alpha, 2).view(-1))

        return loss * alpha_decay_rate
