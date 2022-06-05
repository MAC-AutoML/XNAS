from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.kl import kl_divergence
from torch.distributions.dirichlet import Dirichlet

from xnas.spaces.NASBench201.ops import OPS, ResNetBasicblock, NAS_BENCH_201, channel_shuffle
from xnas.spaces.NASBench201.genos import Structure
from xnas.spaces.NASBench201.cnn import NAS201SearchCell
from .utils import process_step_matrix, prune


class NAS201SearchCell_PartialChannel(NAS201SearchCell):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
        k=4,
    ):
        super(NAS201SearchCell, self).__init__()

        self.k = k
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    xlists = [
                        OPS[op_name](
                            C_in // self.k,
                            C_out // self.k,
                            stride,
                            affine,
                            track_running_stats,
                        )
                        for op_name in op_names
                    ]
                else:
                    xlists = [
                        OPS[op_name](
                            C_in // self.k,
                            C_out // self.k,
                            1,
                            affine,
                            track_running_stats,
                        )
                        for op_name in op_names
                    ]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def MixedOp(self, x, ops, weights):
        dim_2 = x.shape[1]
        xtemp = x[:, : dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k :, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, ops) if not w == 0)
        if self.k == 1:
            return temp1
        ans = torch.cat([temp1, xtemp2], dim=1)
        ans = channel_shuffle(ans, self.k)
        return ans

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    self.MixedOp(x=nodes[j], ops=self.edges[node_str], weights=weights)
                )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    def wider(self, k):
        self.k = k
        for key in self.edges.keys():
            for op in self.edges[key]:
                op.wider(self.in_dim // k, self.out_dim // k)

class TinyNetwork(nn.Module):
    def __init__(
        self,
        C,
        N,
        max_nodes,
        num_classes,
        criterion,
        search_space,
        affine=False,
        track_running_stats=True,
        k=2,
        species="softmax",
        reg_type="l2",
        reg_scale=1e-3,
    ):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self._criterion = criterion
        self.k = k
        self.species = species
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201SearchCell_PartialChannel(
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                    k,
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges and edge2index == cell.edge2index
                    ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._arch_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge, len(search_space))
        )
        self.tau = 10 if species == "gumbel" else None
        self._mask = None

        #### reg
        self.reg_type = reg_type
        self.reg_scale = reg_scale
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.anchor = Dirichlet(torch.ones_like(self._arch_parameters).to(device))

    def _loss(self, input, target):
        logits = self(input)
        loss = self._criterion(logits, target)
        if self.reg_type == "kl":
            loss += self._get_kl_reg()
        return loss

    def _get_kl_reg(self):
        assert self.species == "dirichlet"  # kl implemented only for Dirichlet
        cons = F.elu(self._arch_parameters) + 1
        q = Dirichlet(cons)
        p = self.anchor
        kl_reg = self.reg_scale * torch.sum(kl_divergence(q, p))
        return kl_reg

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def arch_parameters(self):
        return [self._arch_parameters]

    def show_arch_parameters(self, logger):
        with torch.no_grad():
            logger.info(
                "arch-parameters :\n{:}".format(
                    process_step_matrix(
                        self._arch_parameters, "softmax", self._mask
                    ).cpu()
                )
            )
            if self.species == "dirichlet":
                logger.info(
                    "concentration :\n{:}".format(
                        (F.elu(self._arch_parameters) + 1).cpu()
                    )
                )

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def genotype(self):
        genotypes = []
        alphas = process_step_matrix(self._arch_parameters, "softmax", self._mask)
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = alphas[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def pruning(self, num_keep):
        self._mask = prune(self._arch_parameters, num_keep, self._mask)

    def forward(self, inputs):
        alphas = process_step_matrix(
            self._arch_parameters, self.species, self._mask, self.tau
        )

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, NAS201SearchCell_PartialChannel):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

    def wider(self, k):
        self.k = k
        for cell in self.cells:
            if isinstance(cell, NAS201SearchCell_PartialChannel):
                cell.wider(k)


class TinyNetworkGDAS(nn.Module):
    def __init__(
        self,
        C,
        N,
        max_nodes,
        num_classes,
        criterion,
        search_space,
        affine=False,
        track_running_stats=True,
    ):
        super(TinyNetworkGDAS, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self._criterion = criterion
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201SearchCell(
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges and edge2index == cell.edge2index
                    ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._arch_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge, len(search_space))
        )
        self.tau = 10

    def _loss(self, input, target, updateType=None):
        logits = self(input, updateType)
        return self._criterion(logits, target)

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def arch_parameters(self):
        return [self._arch_parameters]

    def show_arch_parameters(self, logger):
        with torch.no_grad():
            logger.info(
                "arch-parameters :\n{:}".format(
                    process_step_matrix(self._arch_parameters, "softmax", None).cpu()
                )
            )

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self._arch_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def forward(self, inputs, updateType=None):
        while True:
            gumbels = -torch.empty_like(self._arch_parameters).exponential_().log()
            logits = (self._arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
            probs = nn.functional.softmax(logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (
                (torch.isinf(gumbels).any())
                or (torch.isinf(probs).any())
                or (torch.isnan(probs).any())
            ):
                continue
            else:
                break

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, NAS201SearchCell):
                feature = cell.forward_gdas(feature, hardwts, index)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits


# build API

def _DrNAS_nb201_CNN(species, criterion):
    from xnas.core.config import cfg
    # if cfg.LOADER.DATASET == 'cifar10':
    return TinyNetwork(
        C=cfg.SPACE.CHANNELS,
        N=cfg.SPACE.LAYERS,
        max_nodes=cfg.SPACE.NODES,
        num_classes=cfg.LOADER.NUM_CLASSES,
        criterion=criterion,
        search_space=NAS_BENCH_201,
        k=cfg.DRNAS.K,
        species=species,
        reg_type=cfg.DRNAS.REG_TYPE,
        reg_scale=cfg.DRNAS.REG_SCALE
    )

def _GDAS_nb201_CNN(criterion):
    from xnas.core.config import cfg
    # if cfg.LOADER.DATASET == 'cifar10':
    return TinyNetworkGDAS(
        C=cfg.SPACE.CHANNELS,
        N=cfg.SPACE.LAYERS,
        max_nodes=cfg.SPACE.NODES,
        num_classes=cfg.LOADER.NUM_CLASSES,
        criterion=criterion,
        search_space=NAS_BENCH_201
    )
