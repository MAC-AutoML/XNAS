import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from xnas.spaces.OFA.utils import (
    min_divisible_value,
    get_same_padding,
    make_divisible,
    drop_connect,
)


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MultiHeadLinearLayer.__name__: MultiHeadLinearLayer,
        ZeroLayer.__name__: ZeroLayer,
        MBConvLayer.__name__: MBConvLayer,
        "MBInvertedConvLayer": MBConvLayer,
        ResidualBlock.__name__: ResidualBlock,
        ResNetBottleneckBlock.__name__: ResNetBottleneckBlock,
    }

    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


"""Activation."""

def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    elif act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    elif act_func == 'swish':
        return MemoryEfficientSwish()
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hswish()"


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hsigmoid()"



class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x

    def __repr__(self):
        return "ShuffleLayer(groups=%d)" % self.groups


class GlobalAvgPool2d(nn.Module):
    def __init__(self, keep_dim=True):
        super(GlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return "GlobalAvgPool2d(keep_dim=%s)" % self.keep_dim


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction

        num_mid = make_divisible(self.channel // self.reduction)

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("expand", nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y

    def __repr__(self):
        return "SE(channel=%d, reduction=%d)" % (self.channel, self.reduction)


class WeightStandardConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization
    https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(WeightStandardConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.WS_EPS = None

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = (
                weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .mean(dim=3, keepdim=True)
            )
            weight = weight - weight_mean
            std = (
                weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
                + self.WS_EPS
            )
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, x):
        if self.WS_EPS is None:
            return super(WeightStandardConv2d, self).forward(x)
        else:
            return F.conv2d(
                x,
                self.weight_standardization(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        return super(WeightStandardConv2d, self).__repr__()[:-1] + ", ws_eps=%s)" % self.WS_EPS
    


# Basic layer to init with Dropout, Conv, BN and activations (order not fixed.)
class MixedDCBRLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(MixedDCBRLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm2d(in_channels)
            else:
                modules["bn"] = nn.BatchNorm2d(out_channels)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(
            self.act_func, self.ops_list[0] != "act" and self.use_bn
        )
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                # dropout before weight operation
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }


class ConvLayer(MixedDCBRLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_se=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_se = use_se

        super(ConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )
        if self.use_se:
            self.add_module("se", SEModule(self.out_channels))

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict(
            {
                "conv": nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    groups=min_divisible_value(self.in_channels, self.groups),
                    bias=self.bias,
                )
            }
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict["shuffle"] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedGroupConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        if self.use_se:
            conv_str = "SE_" + conv_str
        conv_str += "_" + self.act_func.upper()
        if self.use_bn:
            if isinstance(self.bn, nn.GroupNorm):
                conv_str += "_GN%d" % self.bn.num_groups
            elif isinstance(self.bn, nn.BatchNorm2d):
                conv_str += "_BN"
        return conv_str

    @property
    def config(self):
        return {
            "name": ConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "has_shuffle": self.has_shuffle,
            "use_se": self.use_se,
            **super(ConvLayer, self).config,
        }


class IdentityLayer(MixedDCBRLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(IdentityLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return "Identity"

    @property
    def config(self):
        return {
            "name": IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm1d(in_features)
            else:
                modules["bn"] = nn.BatchNorm1d(out_features)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # linear
        modules["weight"] = {
            "linear": nn.Linear(self.in_features, self.out_features, self.bias)
        }

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            "name": LinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


class MultiHeadLinearLayer(nn.Module):
    def __init__(
        self, in_features, out_features, num_heads=1, bias=True, dropout_rate=0
    ):
        super(MultiHeadLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for k in range(num_heads):
            layer = nn.Linear(in_features, out_features, self.bias)
            self.layers.append(layer)

    def forward(self, inputs):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = []
        for layer in self.layers:
            output = layer.forward(inputs)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    @property
    def module_str(self):
        return self.__repr__()

    @property
    def config(self):
        return {
            "name": MultiHeadLinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "num_heads": self.num_heads,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return MultiHeadLinearLayer(**config)

    def __repr__(self):
        return (
            "MultiHeadLinear(in_features=%d, out_features=%d, num_heads=%d, bias=%s, dropout_rate=%s)" % (
                self.in_features,
                self.out_features,
                self.num_heads,
                self.bias,
                self.dropout_rate,
            ))


class ZeroLayer(nn.Module):
    def __init__(self):
        super(ZeroLayer, self).__init__()

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return "Zero"

    @property
    def config(self):
        return {
            "name": ZeroLayer.__name__,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer()


class MBConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func="relu6",
        use_se=False,
        groups=None,
    ):
        super(MBConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.groups = groups

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ("bn", nn.BatchNorm2d(feature_dim)),
                ("act", build_activation(self.act_func, inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        active_groups = (
            feature_dim
            if self.groups is None
            else min_divisible_value(feature_dim, self.groups)
        )
        # assert feature_dim % self.groups == 0
        # active_groups = feature_dim // self.groups
        depth_conv_modules = [
            (
                "conv",
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    pad,
                    groups=active_groups,
                    bias=False,
                ),
            ),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("act", build_activation(self.act_func, inplace=True)),
        ]
        if self.use_se:
            depth_conv_modules.append(("se", SEModule(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = "%dx%d_MBConv%d_%s" % (
            self.kernel_size,
            self.kernel_size,
            expand_ratio,
            self.act_func.upper(),
        )
        if self.use_se:
            layer_str = "SE_" + layer_str
        layer_str += "_O%d" % self.out_channels
        if self.groups is not None:
            layer_str += "_G%d" % self.groups
        if isinstance(self.point_linear.bn, nn.GroupNorm):
            layer_str += "_GN%d" % self.point_linear.bn.num_groups
        elif isinstance(self.point_linear.bn, nn.BatchNorm2d):
            layer_str += "_BN"

        return layer_str

    @property
    def config(self):
        return {
            "name": MBConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
            "groups": self.groups,
        }

    @staticmethod
    def build_from_config(config):
        return MBConvLayer(**config)


class ResidualBlock(nn.Module):
    def __init__(self, conv, shortcut, drop_connect_rate=0):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.mobile_inverted_conv = self.conv   # BigNAS
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        in_channel = x.size(1)
        if self.conv is None or isinstance(self.conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.conv(x)
        else:
            im = self.shortcut(x)
            x = self.conv(x)
            if self.drop_connect_rate > 0 and in_channel == im.size(1) and self.shortcut.reduction == 1:
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            res = x + im
        return res

    @property
    def module_str(self):
        return "(%s, %s)" % (
            self.conv.module_str if self.conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None,
        )

    @property
    def config(self):
        return {
            "name": ResidualBlock.__name__,
            "conv": self.conv.config if self.conv is not None else None,
            "shortcut": self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        conv_config = (
            config["conv"] if "conv" in config else config["mobile_inverted_conv"]
        )
        conv = set_layer_from_config(conv_config)
        shortcut = set_layer_from_config(config["shortcut"])
        return ResidualBlock(conv, shortcut)

    @property
    def mobile_inverted_conv(self):
        return self.conv


class ResNetBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=0.25,
        mid_channels=None,
        act_func="relu",
        groups=1,
        downsample_mode="avgpool_conv",
    ):
        super(ResNetBottleneckBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.groups = groups

        self.downsample_mode = downsample_mode

        if self.mid_channels is None:
            feature_dim = round(self.out_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        feature_dim = make_divisible(feature_dim)
        self.mid_channels = feature_dim

        # build modules
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False),
                    ),
                    ("bn", nn.BatchNorm2d(feature_dim)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        pad = get_same_padding(self.kernel_size)
        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            feature_dim,
                            feature_dim,
                            kernel_size,
                            stride,
                            pad,
                            groups=groups,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(feature_dim)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        self.conv3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(feature_dim, self.out_channels, 1, 1, 0, bias=False),
                    ),
                    ("bn", nn.BatchNorm2d(self.out_channels)),
                ]
            )
        )

        if stride == 1 and in_channels == out_channels:
            self.downsample = IdentityLayer(in_channels, out_channels)
        elif self.downsample_mode == "conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                in_channels, out_channels, 1, stride, 0, bias=False
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(out_channels)),
                    ]
                )
            )
        elif self.downsample_mode == "avgpool_conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg_pool",
                            nn.AvgPool2d(
                                kernel_size=stride,
                                stride=stride,
                                padding=0,
                                ceil_mode=True,
                            ),
                        ),
                        (
                            "conv",
                            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                        ),
                        ("bn", nn.BatchNorm2d(out_channels)),
                    ]
                )
            )
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return "(%s, %s)" % (
            "%dx%d_BottleneckConv_%d->%d->%d_S%d_G%d"
            % (
                self.kernel_size,
                self.kernel_size,
                self.in_channels,
                self.mid_channels,
                self.out_channels,
                self.stride,
                self.groups,
            ),
            "Identity"
            if isinstance(self.downsample, IdentityLayer)
            else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            "name": ResNetBottleneckBlock.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "groups": self.groups,
            "downsample_mode": self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return ResNetBottleneckBlock(**config)


class ShortcutLayer(nn.Module):
    """
        NOTE:
        This class implements similar functionality to `IdentityLayer`, 
        but adds and removes part of the implementation.
    """
    
    def __init__(self, in_channels, out_channels, reduction=1):
        super(ShortcutLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)
        if self.in_channels != self.out_channels:
            x = self.conv(x)
        return x

    @property
    def module_str(self):
        if self.in_channels == self.out_channels and self.reduction == 1:
            conv_str = 'IdentityShortcut'
        else:
            if self.reduction == 1:
                conv_str = '%d-%d_Shortcut' % (self.in_channels, self.out_channels)
            else:
                conv_str = '%d-%d_R%d_Shortcut' % (self.in_channels, self.out_channels, self.reduction)
        return conv_str

    @property
    def config(self):
        return {
            'name': ShortcutLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'reduction': self.reduction,
        }

    @staticmethod
    def build_from_config(config):
        return ShortcutLayer(**config)
