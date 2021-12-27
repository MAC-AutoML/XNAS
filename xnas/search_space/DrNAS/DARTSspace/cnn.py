import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

from xnas.search_space.DARTS.genos import PRIMITIVES, Genotype
from xnas.search_space.DrNAS.DARTSspace.ops import *
from xnas.search_space.DrNAS.utils import process_step_matrix, prune


class Cell(nn.Module):
    def __init__(
        self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, k
    ):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.k = k

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
                op = MixedOp(C, stride, self.k)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)

    def wider(self, k):
        self.k = k
        for op in self._ops:
            op.wider(k)


class NetworkCIFAR(nn.Module):
    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
        k=4,
        reg_type="l2",
        reg_scale=1e-3,
    ):
        super(NetworkCIFAR, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.k = k

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
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
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                k,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

        #### reg
        self.reg_type = reg_type
        self.reg_scale = reg_scale
        self.anchor_normal = Dirichlet(torch.ones_like(self.alphas_normal).cuda())
        self.anchor_reduce = Dirichlet(torch.ones_like(self.alphas_reduce).cuda())

    def new(self):
        model_new = NetworkCIFAR(
            self._C, self._num_classes, self._layers, self._criterion
        ).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def show_arch_parameters(self, logger):
        with torch.no_grad():
            logger.info(
                "alphas normal :\n{:}".format(
                    process_step_matrix(
                        self.alphas_normal, "softmax", self.mask_normal
                    ).cpu()
                )
            )
            logger.info(
                "alphas reduce :\n{:}".format(
                    process_step_matrix(
                        self.alphas_reduce, "softmax", self.mask_reduce
                    ).cpu()
                )
            )
            logger.info(
                "concentration normal:\n{:}".format(
                    (F.elu(self.alphas_normal) + 1).cpu()
                )
            )
            logger.info(
                "concentration reduce:\n{:}".format(
                    (F.elu(self.alphas_reduce) + 1).cpu()
                )
            )

    def pruning(self, num_keep):
        with torch.no_grad():
            self.mask_normal = prune(self.alphas_normal, num_keep, self.mask_normal)
            self.mask_reduce = prune(self.alphas_reduce, num_keep, self.mask_reduce)

    def wider(self, k):
        self.k = k
        for cell in self.cells:
            cell.wider(k)

    def forward(self, input):
        s0 = s1 = self.stem(input)

        weights_normal = process_step_matrix(
            self.alphas_normal, "dirichlet", self.mask_normal
        )
        weights_reduce = process_step_matrix(
            self.alphas_reduce, "dirichlet", self.mask_reduce
        )
        if not self.mask_normal is None:
            assert (weights_normal[~self.mask_normal] == 0.0).all()
        if not self.mask_reduce is None:
            assert (weights_reduce[~self.mask_reduce] == 0.0).all()

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        loss = self._criterion(logits, target)
        if self.reg_type == "kl":
            loss += self._get_kl_reg()
        return loss

    def _get_kl_reg(self):
        cons_normal = F.elu(self.alphas_normal) + 1
        cons_reduce = F.elu(self.alphas_reduce) + 1
        q_normal = Dirichlet(cons_normal)
        q_reduce = Dirichlet(cons_reduce)
        p_normal = self.anchor_normal
        p_reduce = self.anchor_reduce
        kl_reg = self.reg_scale * (
            torch.sum(kl_divergence(q_reduce, p_reduce))
            + torch.sum(kl_divergence(q_normal, p_normal))
        )
        return kl_reg

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
        )
        self.alphas_reduce = Variable(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
        )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        self.mask_normal = None
        self.mask_reduce = None

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                edges = sorted(
                    range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])))
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        # if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        # gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        # gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        gene_normal = _parse(
            process_step_matrix(self.alphas_normal, "softmax", self.mask_normal)
            .data.cpu()
            .numpy()
        )
        gene_reduce = _parse(
            process_step_matrix(self.alphas_reduce, "softmax", self.mask_reduce)
            .data.cpu()
            .numpy()
        )

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype


class NetworkImageNet(nn.Module):
    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
        k=4,
    ):
        super(NetworkImageNet, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.k = k

        C_curr = stem_multiplier * C
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_curr // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                k,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = NetworkImageNet(
            self._C, self._num_classes, self._layers, self._criterion
        ).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def show_arch_parameters(self, logger):
        with torch.no_grad():
            logger.info(
                "alphas normal :\n{:}".format(
                    process_step_matrix(
                        self.alphas_normal, "softmax", self.mask_normal
                    ).cpu()
                )
            )
            logger.info(
                "alphas reduce :\n{:}".format(
                    process_step_matrix(
                        self.alphas_reduce, "softmax", self.mask_reduce
                    ).cpu()
                )
            )
            logger.info(
                "concentration normal:\n{:}".format(
                    (F.elu(self.alphas_normal) + 1).cpu()
                )
            )
            logger.info(
                "concentration reduce:\n{:}".format(
                    (F.elu(self.alphas_reduce) + 1).cpu()
                )
            )

    def pruning(self, num_keep):
        with torch.no_grad():
            self.mask_normal = prune(self.alphas_normal, num_keep, self.mask_normal)
            self.mask_reduce = prune(self.alphas_reduce, num_keep, self.mask_reduce)

    def wider(self, k):
        self.k = k
        for cell in self.cells:
            cell.wider(k)

    def forward(self, input):
        s0 = self.stem0(input)
        s1 = self.stem1(s0)

        weights_normal = process_step_matrix(
            self.alphas_normal, "dirichlet", self.mask_normal
        )
        weights_reduce = process_step_matrix(
            self.alphas_reduce, "dirichlet", self.mask_reduce
        )
        if not self.mask_normal is None:
            assert (weights_normal[~self.mask_normal] == 0.0).all()
        if not self.mask_reduce is None:
            assert (weights_reduce[~self.mask_reduce] == 0.0).all()

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
        )
        self.alphas_reduce = Variable(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
        )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        self.mask_normal = None
        self.mask_reduce = None

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                edges = sorted(
                    range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])))
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        # if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        # gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        # gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        gene_normal = _parse(
            process_step_matrix(self.alphas_normal, "softmax", self.mask_normal)
            .data.cpu()
            .numpy()
        )
        gene_reduce = _parse(
            process_step_matrix(self.alphas_reduce, "softmax", self.mask_reduce)
            .data.cpu()
            .numpy()
        )

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype

# build API

def _DrNASCNN_DARTSspace():
    from xnas.core.config import cfg
    if cfg.SEARCH.DATASET == 'cifar10':
        return NetworkCIFAR(
            C=cfg.SPACE.CHANNEL,
            num_classes=cfg.SEARCH.NUM_CLASSES,
            layers=cfg.SPACE.LAYERS,
            criterion=cfg.SEARCH.LOSS_FUN,
            k=cfg.DRNAS.K,
            reg_type=cfg.DRNAS.REG_TYPE,
            reg_scale=cfg.DRNAS.REG_SCALE
        )
    elif cfg.SEARCH.DATASET == 'imagenet':
        return NetworkImageNet(
            C=cfg.SPACE.CHANNEL,
            num_classes=cfg.SEARCH.NUM_CLASSES,
            layers=cfg.SPACE.LAYERS,
            criterion=cfg.SEARCH.LOSS_FUN,
            k=cfg.DRNAS.K
        )
    else:
        print("dataset not support (cifar10 / imagenet)")
        exit(1)
