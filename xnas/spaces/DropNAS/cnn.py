from xnas.spaces.DARTS.ops import *
import xnas.spaces.DARTS.genos as gt


class Drop_MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights, masks):
        """
        Args:
            x: input
            weights: weight for each operation
            masks: list of boolean
        """
        return sum(w * op(x) for w, op, mask in zip(weights, self._ops, masks) if mask)
        # return sum(w * op(x) for w, op in zip(weights, self._ops))


class DropNASCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

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
            self.preproc0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ReluConvBn(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ReluConvBn(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = Drop_MixedOp(C, C, stride)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag, masks):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list, m_list in zip(self.dag, w_dag, masks):
            s_cur = sum(edges[i](s, w, m) for i, (s, w, m) in enumerate(zip(states, w_list, m_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        return s_out


class DropNASCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = DropNASCell(
                n_nodes=n_nodes, 
                C_pp=C_pp, 
                C_p=C_p, 
                C=C_cur, 
                reduction_p=reduction_p,
                reduction=reduction
            )

            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce, masks_normal, masks_reduce):
        """
        Args:
            weights_xxx: probability contribution of each operation
            masks_xxx: decide whether to drop an operation
        """
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            weights = weights_reduce if cell.reduction else weights_normal
            masks = masks_reduce if cell.reduction else masks_normal
            s0, s1 = s1, cell(s0, s1, weights, masks)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


# build API
def _DropNASCNN():
    from xnas.core.config import cfg
    return DropNASCNN(
        C_in=cfg.SEARCH.INPUT_CHANNELS, 
        C=cfg.SPACE.CHANNELS, 
        n_classes=cfg.LOADER.NUM_CLASSES, 
        n_layers=cfg.SPACE.LAYERS,
        n_nodes=cfg.SPACE.NODES,
    )
