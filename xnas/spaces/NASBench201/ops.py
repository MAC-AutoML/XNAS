"""Basic operations for NAS-Bench-201. Highly copied from DrNAS."""


import torch
import torch.nn as nn

__all__ = ["OPS", "ResNetBasicblock", "SearchSpaceNames"]

OPS = {
    "none": lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, "avg", affine, track_running_stats),
    "max_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, "max", affine, track_running_stats),
    "nor_conv_7x7": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (7, 7), (stride, stride), (3, 3), (1, 1), affine, track_running_stats,),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1), affine, track_running_stats,),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1), affine, track_running_stats,),
    "dua_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1), affine, track_running_stats,),
    "dua_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (5, 5), (stride, stride), (2, 2), (1, 1), affine, track_running_stats,),
    "dil_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(C_in, C_out, (3, 3), (stride, stride), (2, 2), (2, 2), affine, track_running_stats,),
    "dil_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(C_in, C_out, (5, 5), (stride, stride), (4, 4), (2, 2), affine, track_running_stats,),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity()
        if stride == 1 and C_in == C_out
        else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}

NON_PARAMETER_OP = ['none', 'avg_pool_3x3', 'max_pool_3x3', 'skip_connect']
PARAMETER_OP = ['sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 'dil_conv_3x3',
                'dil_conv_5x5', 'conv_7x1_1x7', 'nor_conv_3x3', 'nor_conv_1x1']

CONNECT_NAS_BENCHMARK = ["none", "skip_connect", "nor_conv_3x3"]
NAS_BENCH_201 = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
DARTS_SPACE = [
    "none",
    "skip_connect",
    "dua_sepc_3x3",
    "dua_sepc_5x5",
    "dil_sepc_3x3",
    "dil_sepc_5x5",
    "avg_pool_3x3",
    "max_pool_3x3",
]

SearchSpaceNames = {
    "connect-nas": CONNECT_NAS_BENCHMARK,
    "nas-bench-201": NAS_BENCH_201,
    "darts": DARTS_SPACE,
}

def get_op_index(op_list, parameter_list):
    op_idx_list = []
    for op_idx, op in enumerate(op_list):
        if op in parameter_list:
            op_idx_list.append(op_idx)
    return op_idx_list


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
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


class SepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)


class DualSepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(
            C_in,
            C_in,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats,
        )
        self.op_b = SepConv(
            C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats
        )

    def forward(self, x):
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):
    def __init__(
        self, C_in, C_out, stride, mode, affine=True, track_running_stats=True
    ):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, affine, track_running_stats
            )
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        if self.preprocess:
            self.preprocess.wider(new_C_in, new_C_out)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def wider(self, new_C_in, new_C_out):
        pass


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)

    def wider(self, new_C_in, new_C_out):
        pass


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False)
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)

    def wider(self, new_C_in, new_C_out):
        if self.stride == 2:
            self.convs[0], _ = InChannelWider(self.convs[0], new_C_in)
            self.convs[0], index1 = OutChannelWider(self.convs[0], new_C_out // 2)
            self.convs[1], _ = InChannelWider(self.convs[1], new_C_in)
            self.convs[1], index2 = OutChannelWider(
                self.convs[1], new_C_out - new_C_out // 2
            )
            self.bn, _ = BNWider(self.bn, new_C_out, index=torch.cat([index1, index2]))
        elif self.stride == 1:
            self.conv, _ = InChannelWider(self.conv, new_C_in)
            self.conv, index = OutChannelWider(self.conv, new_C_out)
            self.bn, _ = BNWider(self.bn, new_C_out, index=index)


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
