import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from xnas.search_space.utils import build_activation, SEModule, make_divisible, get_same_padding
from xnas.core.config import cfg


class ConvLayer(nn.Module):
    """Standard convolution layers with CBR"""

    def __init__(self, w_in, w_out, kernel_size, stride, padding, conv_act):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv_act = build_activation(conv_act)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class LinearLayer(nn.Module):
    """Standard linear layers"""

    def __init__(self, in_features, out_features, act_func=None, dropout_rate=0, bias=True):
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act = build_activation(act_func) if act_func is not None else None
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        x = self.act(x) if self.act is not None else x
        return x


class MBHead(nn.Module):
    """MobileNetV2/V3 head: generate by input."""

    def __init__(self, w_in, head_channels, head_acts, nc):
        super(MBHead, self).__init__()
        assert len(head_channels) == len(head_channels)
        self.conv = nn.Conv2d(
            w_in, head_channels[0], 1, stride=1, padding=0, bias=False)
        self.conv_bn = nn.BatchNorm2d(
            head_channels[0], eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv_act = build_activation(head_acts[0])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        _head_acts = head_acts[1:]
        _head_channels = head_channels[1:]
        self.linears = []
        pre_w = head_channels[0]
        for i, act in enumerate(_head_acts):
            self.linears.append(nn.Linear(pre_w, _head_channels[i]))
            # self.linears.append(nn.BatchNorm1d(_head_channels[i]))
            self.linears.append(build_activation(act))
            pre_w = _head_channels[i]
        if len(self.linears) > 0:
            self.linears = nn.Sequential(*self.linears)
        if cfg.MB.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.MB.DROPOUT_RATIO)
        self.fc = nn.Linear(head_channels[-1], nc, bias=True)

    def forward(self, x):
        x = self.conv_act(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if len(self.linears) > 0:
            x = self.linears(x)
        x = self.dropout(x) if hasattr(self, "dropout") else x
        x = self.fc(x)
        return x


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv).
    """

    def __init__(self, in_channel, expand_ratio, kernel_size, stride, act_func, se, out_channel):
        # expansion, 3x3 dwise, BN, Swish, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        middle_channel = int(in_channel * expand_ratio)
        middle_channel = make_divisible(middle_channel, 8)
        if middle_channel != in_channel:
            self.expand = True
            self.inverted_bottleneck_conv = nn.Conv2d(in_channel, middle_channel, 1, stride=1,
                                                      padding=0, bias=False)
            self.inverted_bottleneck_bn = nn.BatchNorm2d(
                middle_channel, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
            self.inverted_bottleneck_act = build_activation(act_func)
        else:
            self.expand = False
        self.depth_conv = nn.Conv2d(middle_channel, middle_channel, kernel_size,
                                    stride=stride, groups=middle_channel, padding=get_same_padding(kernel_size), bias=False)
        self.depth_bn = nn.BatchNorm2d(
            middle_channel, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.depth_act = build_activation(act_func)
        if se > 0:
            self.depth_se = SEModule(middle_channel, se)
        self.point_linear_conv = nn.Conv2d(
            middle_channel, out_channel, 1, stride=1, padding=0, bias=False)
        self.point_linear_bn = nn.BatchNorm2d(
            out_channel, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = stride == 1 and in_channel == out_channel

    def forward(self, x):
        f_x = x
        if self.expand:
            f_x = self.inverted_bottleneck_act(
                self.inverted_bottleneck_bn(self.inverted_bottleneck_conv(f_x)))
        f_x = self.depth_act(self.depth_bn(self.depth_conv(f_x)))
        if hasattr(self, 'depth_se'):
            f_x = self.depth_se(f_x)
        f_x = self.point_linear_bn(self.point_linear_conv(f_x))
        if self.has_skip:
            f_x = x + f_x
        return f_x


class ResidualBlock(nn.Module):
    def __init__(self, conv=None, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.conv is None:
            return x
        elif self.shortcut is None:
            return self.conv(x)
        else:
            return self.conv(x) + self.shortcut(x)
