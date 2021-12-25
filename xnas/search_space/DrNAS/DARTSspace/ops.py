import torch
import torch.nn as nn
from xnas.search_space.DARTS.genos import PRIMITIVES
from xnas.search_space.DrNAS.net2wider import InChannelWider, OutChannelWider, BNWider


OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    # 'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    # 'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "avg_pool_3x3": lambda C, stride, affine: AvgPoolBN(C, stride=stride),
    "max_pool_3x3": lambda C, stride, affine: MaxPoolBN(C, stride=stride),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(
        C, C, 5, stride, 2, affine=affine
    ),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(
        C, C, 7, stride, 3, affine=affine
    ),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine
    ),
    "conv_7x1_1x7": lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
}

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MixedOp(nn.Module):
    def __init__(self, C, stride, k):
        super(MixedOp, self).__init__()
        self.k = k
        self.C = C
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)

        for primitive in PRIMITIVES:
            op = OPS[primitive](C // self.k, stride, False)
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


class AvgPoolBN(nn.Module):
    def __init__(self, C_out, stride):
        super(AvgPoolBN, self).__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
            nn.BatchNorm2d(C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        bn = self.op[1]
        bn, _ = BNWider(bn, new_C_out)
        self.op[1] = bn


class MaxPoolBN(nn.Module):
    def __init__(self, C_out, stride):
        super(MaxPoolBN, self).__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.BatchNorm2d(C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        bn = self.op[1]
        bn, _ = BNWider(bn, new_C_out)
        self.op[1] = bn


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
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


class DilConv(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
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


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        conv1 = self.op[1]
        conv2 = self.op[2]
        conv3 = self.op[5]
        conv4 = self.op[6]
        bn1 = self.op[3]
        bn2 = self.op[7]
        conv1, index = OutChannelWider(conv1, new_C_out)
        conv1.groups = new_C_in
        conv2, _ = InChannelWider(conv2, new_C_in, index=index)
        conv2, index = OutChannelWider(conv2, new_C_out)
        bn1, _ = BNWider(bn1, new_C_out, index=index)

        conv3, index = OutChannelWider(conv3, new_C_out)
        conv3.groups = new_C_in
        conv4, _ = InChannelWider(conv4, new_C_in, index=index)
        conv4, index = OutChannelWider(conv4, new_C_out)
        bn2, _ = BNWider(bn2, new_C_out, index=index)
        self.op[1] = conv1
        self.op[2] = conv2
        self.op[5] = conv3
        self.op[6] = conv4
        self.op[3] = bn1
        self.op[7] = bn2


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def wider(self, new_C_in, new_C_out):
        pass


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)

    def wider(self, new_C_in, new_C_out):
        pass


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def wider(self, new_C_in, new_C_out):
        self.conv_1, _ = InChannelWider(self.conv_1, new_C_in)
        self.conv_1, index1 = OutChannelWider(self.conv_1, new_C_out // 2)
        self.conv_2, _ = InChannelWider(self.conv_2, new_C_in)
        self.conv_2, index2 = OutChannelWider(self.conv_2, new_C_out // 2)
        self.bn, _ = BNWider(self.bn, new_C_out, index=torch.cat([index1, index2]))
