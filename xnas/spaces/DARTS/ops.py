import torch
import torch.nn as nn
import torch.nn.functional as F


OPS = {
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
    'nor_conv_3x3': lambda C_in, C_out, stride, affine: ReluConvBn(C_in, C_out, 3, stride, 1, affine=affine),
    'nor_conv_1x1': lambda C_in, C_out, stride, affine: ReluConvBn(C_in, C_out, 1, stride, 0, affine=affine),
}

NON_PARAMETER_OP = ['none', 'avg_pool_3x3', 'max_pool_3x3', 'skip_connect']
PARAMETER_OP = ['sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 'dil_conv_3x3',
                'dil_conv_5x5', 'conv_7x1_1x7', 'nor_conv_3x3', 'nor_conv_1x1']

NAS_BENCH_201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
DARTS_SPACE = ['none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'avg_pool_3x3', 'max_pool_3x3']


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
            op = OPS[primitive](C_in, C_out, stride, affine=False)
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


class DrNAS_MixedOp(nn.Module):
    def __init__(self, C, stride, k, basic_op_list):
        super(DrNAS_MixedOp, self).__init__()
        self.k = k
        self.C = C
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)

        for primitive in basic_op_list:
            op = OPS[primitive](C // self.k, C // self.k, stride, False)
            # if 'pool' in primitive:
            #   op = nn.Sequential(op, nn.BatchNorm2d(C //self.k, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        dim_2 = x.shape[1]
        xtemp = x[:, : dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k :, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops) if not w == 0)
        if self.k == 1:
            return temp1
        # reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, self.k)
        # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans

    def wider(self, k):
        self.k = k
        for op in self._ops:
            op.wider(self.C // k, self.C // k)


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
        self.conv_a = ReluConvBn(inplanes, planes, 3, stride, 1)
        self.conv_b = ReluConvBn(planes, planes, 3, 1, 1)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReluConvBn(inplanes, planes, 1, 1, 0)
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


# ----------------------------------------- #

class ConvBnRelu(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding=1):
        super(ConvBnRelu, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(C_out, affine=True, momentum=0.997, eps=1e-5),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class Conv3x3BnRelu(nn.Module):
    def __init__(self, channels, stride):
        super(Conv3x3BnRelu, self).__init__()
        self.op = ConvBnRelu(
            C_in=channels, C_out=channels, kernel_size=3, stride=stride
        )

    def forward(self, x):
        return self.op(x)


class Conv1x1BnRelu(nn.Module):
    def __init__(self, channels, stride):
        super(Conv1x1BnRelu, self).__init__()
        self.op = ConvBnRelu(
            C_in=channels, C_out=channels, kernel_size=1, stride=stride, padding=0
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """(Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        conv1 = self.op[1]
        conv2 = self.op[2]
        bn = self.op[3]
        conv1, index = OutChannelWider(conv1, new_C_out)
        conv1.groups = new_C_in
        conv2, _ = InChannelWider(conv2, new_C_in, index=index)
        conv2, index = OutChannelWider(conv2, new_C_out)
        bn, _ = BNWider(bn, new_C_out, index=index)
        self.op[1] = conv1
        self.op[2] = conv2
        self.op[3] = bn


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)
    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.0):
        """[!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return "p={}, inplace".format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class FacConv(nn.Module):
    """Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0, "C_out should be even!"
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def wider(self, new_C_in, new_C_out):
        self.conv_1, _ = InChannelWider(self.conv_1, new_C_in)
        self.conv_1, index1 = OutChannelWider(self.conv_1, new_C_out // 2)
        self.conv_2, _ = InChannelWider(self.conv_2, new_C_in)
        self.conv_2, index2 = OutChannelWider(self.conv_2, new_C_out // 2)
        self.bn, _ = BNWider(self.bn, new_C_out, index=torch.cat([index1, index2]))


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def wider(self, new_C_in, new_C_out):
        pass


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == "max":
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == "avg":
            self.pool = nn.AvgPool2d(
                kernel_size, stride, padding, count_include_pad=False
            )
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out

    def wider(self, new_C_in, new_C_out):
        self.bn, _ = BNWider(self.bn, new_C_out)


class SepConv(nn.Module):
    """Depthwise separable conv
    DilConv(dilation=1) * 2
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            DilConv(
                C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine
            ),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        self.op[0] = self.op[0].wider(new_C_in, new_C_out)
        self.op[1] = self.op[1].wider(new_C_in, new_C_out)
        

class ReluConvBn(nn.Module):
    """Standard conv
    ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)
    
    def wider(self, new_C_in, new_C_out):
        conv = self.op[1]
        bn = self.op[2]
        conv, _ = InChannelWider(conv, new_C_in)
        conv, index = OutChannelWider(conv, new_C_out)
        bn, _ = BNWider(bn, new_C_out, index=index)
        self.op[1] = conv
        self.op[2] = bn


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.0
        # re-sizing by stride
        return x[:, :, :: self.stride, :: self.stride] * 0.0

    def wider(self, new_C_in, new_C_out):
        pass



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


# bias = 0
def InChannelWider(module, new_channels, index=None):
    weight = module.weight
    in_channels = weight.size(1)

    if index is None:
        index = torch.randint(
            low=0, high=in_channels, size=(new_channels - in_channels,)
        )
    module.weight = nn.Parameter(
        torch.cat([weight, weight[:, index, :, :].clone()], dim=1), requires_grad=True
    )

    module.in_channels = new_channels
    module.weight.in_index = index
    module.weight.t = "conv"
    if hasattr(weight, "out_index"):
        module.weight.out_index = weight.out_index
    module.weight.raw_id = weight.raw_id if hasattr(weight, "raw_id") else id(weight)
    return module, index


# bias = 0
def OutChannelWider(module, new_channels, index=None):
    weight = module.weight
    out_channels = weight.size(0)

    if index is None:
        index = torch.randint(
            low=0, high=out_channels, size=(new_channels - out_channels,)
        )
    module.weight = nn.Parameter(
        torch.cat([weight, weight[index, :, :, :].clone()], dim=0), requires_grad=True
    )

    module.out_channels = new_channels
    module.weight.out_index = index
    module.weight.t = "conv"
    if hasattr(weight, "in_index"):
        module.weight.in_index = weight.in_index
    module.weight.raw_id = weight.raw_id if hasattr(weight, "raw_id") else id(weight)
    return module, index


def BNWider(module, new_features, index=None):
    running_mean = module.running_mean
    running_var = module.running_var
    if module.affine:
        weight = module.weight
        bias = module.bias
    num_features = module.num_features

    if index is None:
        index = torch.randint(
            low=0, high=num_features, size=(new_features - num_features,)
        )
    module.running_mean = torch.cat([running_mean, running_mean[index].clone()])
    module.running_var = torch.cat([running_var, running_var[index].clone()])
    if module.affine:
        module.weight = nn.Parameter(
            torch.cat([weight, weight[index].clone()], dim=0), requires_grad=True
        )
        module.bias = nn.Parameter(
            torch.cat([bias, bias[index].clone()], dim=0), requires_grad=True
        )

        module.weight.out_index = index
        module.bias.out_index = index
        module.weight.t = "bn"
        module.bias.t = "bn"
        module.weight.raw_id = (
            weight.raw_id if hasattr(weight, "raw_id") else id(weight)
        )
        module.bias.raw_id = bias.raw_id if hasattr(bias, "raw_id") else id(bias)
    module.num_features = new_features
    return module, index
