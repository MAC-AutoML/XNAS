from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from xnas.spaces.OFA.utils import (
    get_same_padding, 
    val2list, 
    make_divisible,
)
from xnas.spaces.OFA.ops import (
    build_activation,
    set_layer_from_config,
    SEModule,
    WeightStandardConv2d,
    MBConvLayer,
    ConvLayer,
    IdentityLayer,
    ResNetBottleneckBlock,
    LinearLayer,
)


class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1, kernel_trans=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        self.KERNEL_TRANSFORM_MODE = kernel_trans

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # [!] 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small ** 2), requires_grad=True
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :out_channel, :in_channel, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks ** 2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, WeightStandardConv2d)
            else filters
        )
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, in_channel)
        return y


class DynamicConv2d(nn.Module):
    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        padding = get_same_padding(self.kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, WeightStandardConv2d)
            else filters
        )
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicGroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size_list,
        groups_list,
        stride=1,
        dilation=1,
    ):
        super(DynamicGroupConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_list = kernel_size_list
        self.groups_list = groups_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=min(self.groups_list),
            bias=False,
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_groups = min(self.groups_list)

    def get_active_filter(self, kernel_size, groups):
        start, end = sub_filter_start_end(max(self.kernel_size_list), kernel_size)
        filters = self.conv.weight[:, :, start:end, start:end]

        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_in_channels = self.in_channels // groups
        sub_ratio = filters.size(1) // sub_in_channels

        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_in_channels
            filter_crops.append(sub_filter[:, start : start + sub_in_channels, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def forward(self, x, kernel_size=None, groups=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if groups is None:
            groups = self.active_groups

        filters = self.get_active_filter(kernel_size, groups).contiguous()
        padding = get_same_padding(kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, WeightStandardConv2d)
            else filters
        )
        y = F.conv2d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            groups,
        )
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicGroupNorm(nn.GroupNorm):
    def __init__(
        self, num_groups, num_channels, eps=1e-5, affine=True, channel_per_group=None
    ):
        super(DynamicGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.channel_per_group = channel_per_group

    def forward(self, x):
        n_channels = x.size(1)
        n_groups = n_channels // self.channel_per_group
        return F.group_norm(
            x, n_groups, self.weight[:n_channels], self.bias[:n_channels], self.eps
        )

    @property
    def bn(self):
        return self


class DynamicSE(SEModule):
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def get_active_reduce_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.reduce.weight[:num_mid, :in_channel, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.reduce.weight[:num_mid, :, :, :], groups, dim=1
            )
            return torch.cat(
                [sub_filter[:, :sub_in_channels, :, :] for sub_filter in sub_filters],
                dim=1,
            )

    def get_active_reduce_bias(self, num_mid):
        return (
            self.fc.reduce.bias[:num_mid] if self.fc.reduce.bias is not None else None
        )

    def get_active_expand_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.expand.weight[:in_channel, :num_mid, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.expand.weight[:, :num_mid, :, :], groups, dim=0
            )
            return torch.cat(
                [sub_filter[:sub_in_channels, :, :, :] for sub_filter in sub_filters],
                dim=0,
            )

    def get_active_expand_bias(self, in_channel, groups=None):
        if groups is None or groups == 1:
            return (
                self.fc.expand.bias[:in_channel]
                if self.fc.expand.bias is not None
                else None
            )
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_bias_list = torch.chunk(self.fc.expand.bias, groups, dim=0)
            return torch.cat(
                [sub_bias[:sub_in_channels] for sub_bias in sub_bias_list], dim=0
            )

    def forward(self, x, groups=None):
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction)

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_filter = self.get_active_reduce_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        reduce_bias = self.get_active_reduce_bias(num_mid)
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_filter = self.get_active_expand_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        expand_bias = self.get_active_expand_bias(in_channel, groups=groups)
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y


class DynamicLinearLayer(nn.Module):
    def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0):
        super(DynamicLinearLayer, self).__init__()

        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list),
            max_out_features=self.out_features,
            bias=self.bias,
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return "DyLinear(%d, %d)" % (max(self.in_features_list), self.out_features)

    @property
    def config(self):
        return {
            "name": DynamicLinear.__name__,
            "in_features_list": self.in_features_list,
            "out_features": self.out_features,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        sub_layer = LinearLayer(
            in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate
        )
        sub_layer = sub_layer.to(self.parameters().__next__().device)
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_(
            self.linear.get_active_weight(self.out_features, in_features).data
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(
                self.linear.get_active_bias(self.out_features).data
            )
        return sub_layer

    def get_active_subnet_config(self, in_features):
        return {
            "name": LinearLayer.__name__,
            "in_features": in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }


class DynamicMBConvLayer(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        expand_ratio_list=6,
        stride=1,
        act_func="relu6",
        use_se=False,
        kernel_trans=1,
    ):
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = val2list(kernel_size_list)
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se

        # build modules
        max_middle_channel = make_divisible(round(max(self.in_channel_list) * max(self.expand_ratio_list)))
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                    ("conv", DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
                    ("conv", DynamicSeparableConv2d(max_middle_channel, self.kernel_size_list, stride=self.stride, kernel_trans=kernel_trans)),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func)),
                ]
            )
        )
        if self.use_se:
            self.depth_conv.add_module("se", DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(max_middle_channel, max(self.out_channel_list)),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = make_divisible(
                round(in_channel * self.active_expand_ratio))

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.use_se:
            return "SE(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )
        else:
            return "(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )

    @property
    def config(self):
        return {
            "name": DynamicMBConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "expand_ratio_list": self.expand_ratio_list,
            "stride": self.stride,
            "act_func": self.act_func,
            "use_se": self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    def active_middle_channel(self, in_channel):
        return make_divisible(round(in_channel * self.active_expand_ratio))

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(self.parameters().__next__().device)
        if not preserve_weight:
            return sub_layer

        middle_channel = self.active_middle_channel(in_channel)
        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.get_active_filter(
                    middle_channel, in_channel
                ).data,
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(
                middle_channel, self.active_kernel_size
            ).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(
                middle_channel // SEModule.REDUCTION,
            )
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.get_active_reduce_weight(se_mid, middle_channel).data
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.get_active_reduce_bias(se_mid).data
            )

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.get_active_expand_weight(se_mid, middle_channel).data
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.get_active_expand_bias(middle_channel).data
            )

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.get_active_filter(
                self.active_out_channel, middle_channel
            ).data
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            "name": MBConvLayer.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.active_kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_middle_channel(in_channel),
            "act_func": self.act_func,
            "use_se": self.use_se,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = torch.sum(
            torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3)
        )
        if isinstance(self.depth_conv.bn, DynamicGroupNorm):
            channel_per_group = self.depth_conv.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.in_channel_list) * expand))
                for expand in sorted_expand_list
            ]

            right = len(importance)
            base = -len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            # se expand: output dim 0 reorganize
            se_expand = self.depth_conv.se.fc.expand
            se_expand.weight.data = torch.index_select(
                se_expand.weight.data, 0, sorted_idx
            )
            se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
            # se reduce: input dim 1 reorganize
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(
                se_reduce.weight.data, 1, sorted_idx
            )
            # middle weight reorganize
            se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)

        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx


class DynamicConvLayer(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bn=True,
        act_func="relu6",
    ):
        super(DynamicConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self):
        return "DyConv(O%d, K%d, S%d)" % (
            self.active_out_channel,
            self.kernel_size,
            self.stride,
        )

    @property
    def config(self):
        return {
            "name": DynamicConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(self.parameters().__next__().device)

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(
            self.conv.get_active_filter(self.active_out_channel, in_channel).data
        )
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            "name": ConvLayer.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }


class DynamicResNetBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        expand_ratio_list=0.25,
        kernel_size=3,
        stride=1,
        act_func="relu",
        downsample_mode="avgpool_conv",
    ):
        super(DynamicResNetBottleneckBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)))

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(max(self.in_channel_list), max_middle_channel),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(
                            max_middle_channel, max_middle_channel, kernel_size, stride
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        self.conv3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(max_middle_channel, max(self.out_channel_list)),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = IdentityLayer(
                max(self.in_channel_list), max(self.out_channel_list)
            )
        elif self.downsample_mode == "conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            DynamicConv2d(
                                max(self.in_channel_list),
                                max(self.out_channel_list),
                                stride=stride,
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
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
                            DynamicConv2d(
                                max(self.in_channel_list), max(self.out_channel_list)
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                    ]
                )
            )
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = feature_dim
        self.conv3.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = self.active_out_channel

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
            "%dx%d_BottleneckConv_in->%d->%d_S%d"
            % (
                self.kernel_size,
                self.kernel_size,
                self.active_middle_channels,
                self.active_out_channel,
                self.stride,
            ),
            "Identity"
            if isinstance(self.downsample, IdentityLayer)
            else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            "name": DynamicResNetBottleneckBlock.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "expand_ratio_list": self.expand_ratio_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "act_func": self.act_func,
            "downsample_mode": self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicResNetBottleneckBlock(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim)
        return feature_dim

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(self.parameters().__next__().device)
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(
                self.active_middle_channels, in_channel
            ).data
        )
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(
                self.active_middle_channels, self.active_middle_channels
            ).data
        )
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        sub_layer.conv3.conv.weight.data.copy_(
            self.conv3.conv.get_active_filter(
                self.active_out_channel, self.active_middle_channels
            ).data
        )
        copy_bn(sub_layer.conv3.bn, self.conv3.bn.bn)

        if not isinstance(self.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(
                    self.active_out_channel, in_channel
                ).data
            )
            copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            "name": ResNetBottleneckBlock.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_middle_channels,
            "act_func": self.act_func,
            "groups": 1,
            "downsample_mode": self.downsample_mode,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        # conv3 -> conv2
        importance = torch.sum(
            torch.abs(self.conv3.conv.conv.weight.data), dim=(0, 2, 3)
        )
        if isinstance(self.conv2.bn, DynamicGroupNorm):
            channel_per_group = self.conv2.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.out_channel_list) * expand))
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = -len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.conv3.conv.conv.weight.data = torch.index_select(
            self.conv3.conv.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.conv2.bn.bn, sorted_idx)
        self.conv2.conv.conv.weight.data = torch.index_select(
            self.conv2.conv.conv.weight.data, 0, sorted_idx
        )

        # conv2 -> conv1
        importance = torch.sum(
            torch.abs(self.conv2.conv.conv.weight.data), dim=(0, 2, 3)
        )
        if isinstance(self.conv1.bn, DynamicGroupNorm):
            channel_per_group = self.conv1.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.out_channel_list) * expand))
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = -len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

        self.conv2.conv.conv.weight.data = torch.index_select(
            self.conv2.conv.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv.weight.data = torch.index_select(
            self.conv1.conv.conv.weight.data, 0, sorted_idx
        )

        return None


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    feature_dim = (
        target_bn.num_channels
        if isinstance(target_bn, nn.GroupNorm)
        else target_bn.num_features
    )

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])
