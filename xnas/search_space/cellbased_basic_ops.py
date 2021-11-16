from xnas.search_space.cellbased_layers import *


OPS_ = {
    'none': lambda C_in, C_out, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine: PoolBN('avg', C_in, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C_in, C_out, stride, affine: PoolBN('max', C_in, 3, stride, 1, affine=affine),
    'skip_connect': lambda C_in, C_out, stride, affine: Identity() if stride == 1 else FactorizedReduce(C_in, C_out, affine=affine),
    'sep_conv_3x3': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 7, stride, 3, affine=affine),
    # 5x5
    'dil_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine=affine),
    # 9x9
    'dil_conv_5x5': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C_in, C_out, stride, affine: FacConv(C_in, C_out, 7, stride, 3, affine=affine),
    'nor_conv_3x3': lambda C_in, C_out, stride, affine: StdConv(C_in, C_out, 3, stride, 1, affine=affine),
    'nor_conv_1x1': lambda C_in, C_out, stride, affine: StdConv(C_in, C_out, 1, stride, 0, affine=affine),
}

NON_PARAMETER_OP = ['none', 'avg_pool_3x3', 'max_pool_3x3', 'skip_connect']
PARAMETER_OP = ['sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 'dil_conv_3x3',
                'dil_conv_5x5', 'conv_7x1_1x7', 'nor_conv_3x3', 'nor_conv_1x1']


def get_op_index(op_list, parameter_list):
    op_idx_list = []
    for op_idx, op in enumerate(op_list):
        if op in parameter_list:
            op_idx_list.append(op_idx)
    return op_idx_list


def darts_weight_unpack(weight, n_nodes, input_nodes=2):
    w_dag = []
    start_index = 0
    end_index = input_nodes
    for i in range(n_nodes):
        w_dag.append(weight[start_index:end_index])
        start_index = end_index
        end_index += input_nodes + i + 1
    return w_dag


def parse_from_numpy(alpha, k, basic_op_list=None):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert basic_op_list[-1] == 'none'  # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(
            torch.tensor(edges[:, :-1]), 1)  # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = basic_op_list[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


class MixedOp(nn.Module):
    """ define the basic search space operation according to string """

    def __init__(self, C_in, C_out, stride, basic_op_list=None):
        super().__init__()
        self._ops = nn.ModuleList()
        assert basic_op_list is not None, "the basic op list cannot be none!"
        basic_primitives = basic_op_list
        for primitive in basic_primitives:
            op = OPS_[primitive](C_in, C_out, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        assert len(self._ops) == len(weights)
        _x = []
        for i, value in enumerate(weights):
            if value == 1:
                _x.append(self._ops[i](x))
            if 0 < value < 1:
                _x.append(value * self._ops[i](x))
        return sum(_x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = StdConv(inplanes, planes, 3, stride, 1)
        self.conv_b = StdConv(planes, planes, 3, 1, 1)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = StdConv(inplanes, planes, 1, 1, 0)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(
            name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock
