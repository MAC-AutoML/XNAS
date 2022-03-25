from xnas.search_space.cellbased_basic_ops import *
import xnas.search_space.cellbased_basic_genotypes as gt

basic_op_list = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none']

# Augmented DARTS

def geno_from_alpha(theta):
    Genotype = namedtuple(
        'Genotype', 'normal normal_concat reduce reduce_concat')
    theta_norm = darts_weight_unpack(
        theta[0:14], 4)
    theta_reduce = darts_weight_unpack(
        theta[14:], 4)
    gene_normal = parse_from_numpy(
        theta_norm, k=2, basic_op_list=basic_op_list)
    gene_reduce = parse_from_numpy(
        theta_reduce, k=2, basic_op_list=basic_op_list)
    concat = range(2, 6)  # concat all intermediate nodes
    return Genotype(normal=gene_normal, normal_concat=concat,
                    reduce=gene_reduce, reduce_concat=concat)

def reformat_DARTS(genotype):
    """
    format genotype for DARTS-like
    from:
        Genotype(normal=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 0)], [('sep_conv_3x3', 2), ('max_pool_3x3', 1)], [('sep_conv_3x3', 3), ('dil_conv_3x3', 2)], [('dil_conv_5x5', 4), ('dil_conv_5x5', 3)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1)], [('max_pool_3x3', 0), ('dil_conv_5x5', 2)], [('max_pool_3x3', 0), ('sep_conv_5x5', 1)], [('dil_conv_5x5', 4), ('max_pool_3x3', 0)]], reduce_concat=range(2, 6))
    to:
        Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
    """
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    _normal = []
    _reduce = []
    for i in genotype.normal:
        for j in i:
            _normal.append(j)
    for i in genotype.reduce:
        for j in i:
            _reduce.append(j)
    _normal_concat = [i for i in genotype.normal_concat]
    _reduce_concat = [i for i in genotype.reduce_concat]
    r_genotype = Genotype(
        normal=_normal,
        normal_concat=_normal_concat,
        reduce=_reduce,
        reduce_concat=_reduce_concat
    )
    return r_genotype

class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        # assert input_size in [7, 8]
        super().__init__()
        if input_size in [7, 8]:
            self.net = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=input_size-5, padding=0, count_include_pad=False), # 2x2 out
                nn.Conv2d(C, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True))
        else:
            self.net = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Conv2d(C, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, kernel_size=2, bias=False),  # 1x1 out
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True))
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """
    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal)

        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C)
        else:
            self.preproc0 = StdConv(C_pp, C, 1, 1, 0)
        self.preproc1 = StdConv(C_p, C, 1, 1, 0)

        # generate dag
        if reduction:
            gene = genotype.reduce
            self.concat = genotype.reduce_concat
        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat

        self.dag = gt.to_dag(C, gene, reduction)

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)

        return s_out


class AugmentCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 stem_multiplier=3):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
#         self.genotype = gt.from_str(genotype)
        self.genotype = genotype
        # aux head position
        self.aux_pos = 2*n_layers//3 if auxiliary else -1

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(self.genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

            if i == self.aux_pos:
                # [!] this auxiliary head is ignored in computing parameter size
                #     by the name 'aux_head'
                self.aux_head = AuxiliaryHead(input_size//4, C_p, n_classes)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        features = []
        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i in [int(self.n_layers//3-1), int(2*self.n_layers//3-1), int(self.n_layers-1)]:
                features.append(s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)
        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        
        return features, logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p
