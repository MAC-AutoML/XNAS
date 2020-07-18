# This code is highly referenced from onece-for-all from https://github.com/mit-han-lab/once-for-all
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import OrderedDict

import xnas.core.logging as logging
from xnas.search_space.utils import (SEModule, build_activation,
                                     get_same_padding, make_divisible,
                                     sub_filter_start_end)


logger = logging.get_logger(__name__)


class DynamicSeparableConv2d(nn.Module):
    """
    DynamicseparableConv2D is a separable convolution operations with dynamic act, kernel size and input channels
    in_channel_list: type list example -> [24, 32, 68, xxx]
    kernel_size_list: type list example -> [3, 5, 7]
    act_list: type list example -> ['relu6', 'swish']
    weight_sharing_mode: type int example -> 0
        if weight_sharing_mode == 0:
            all the weight are from 1 big tensor e.g. 68x7x7
        if weight_sharing_mode == 1:
            all the weight are from 1 big tensor, e.g. 68x7x7, for different kernel size it will transform with matrix multi,
            which is identical with https://github.com/mit-han-lab/once-for-all
        if weight_sharing_mode == 2:
            the weight of different kernels have different weight tensors, for example, if kernel_size_list=[3, 5, 7] and
            in_channel_list=[24, 32, 68], we have 3 weight tensors, 68x3x3, 68x5x5, 68x7x7
        if weight_sharing_mode == 3:
            the weight of different kernels and different channels have different weight tensors
    """

    def __init__(self, in_channel_list, kernel_size_list, weight_sharing_mode=0, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()
        assert weight_sharing_mode > 3, "The weight sharing mode should be less than 3"
        in_channel_list = in_channel_list.sort()
        kernel_size_list = kernel_size_list.sort()
        self.in_channel_list = in_channel_list
        self.kernel_size_list = kernel_size_list
        self.max_in_channels = max(in_channel_list)
        self.max_kernel_size = max(kernel_size_list)
        self.stride = stride
        self.dilation = dilation
        self.weight_sharing_mode = weight_sharing_mode
        self.active_kernel = self.max_kernel_size

        if self.weight_sharing_mode == 0 or self.weight_sharing_mode == 1:
            # all in one
            self.conv = nn.Conv2d(
                self.max_in_channels, self.max_in_channels, self.max_kernel_size, self.stride,
                groups=self.max_in_channels, bias=False,
            )
            if self.weight_sharing_mode == 1:
                # register scaling parameters
                # 7to5_matrix, 5to3_matrix
                scale_params = {}
                for i in range(len(self.kernel_size_list) - 1):
                    ks_small = self.kernel_size_list[i]
                    ks_larger = self.kernel_size_list[i + 1]
                    param_name = '%dto%d' % (ks_larger, ks_small)
                    scale_params['%s_matrix' % param_name] = Parameter(
                        torch.eye(ks_small ** 2))
                for name, param in scale_params.items():
                    self.register_parameter(name, param)
        elif self.weight_sharing_mode == 2:
            # do not share the weight in different kernel
            self.conv = nn.ModuleDict()
            for kernel in self.kernel_size_list:
                self.conv[str(kernel)] = nn.Conv2d(
                    self.max_in_channels, self.max_in_channels, kernel, self.stride,
                    groups=self.max_in_channels, bias=False,
                )
        else:
            number_of_candidates = len(
                self.in_channel_list) * len(self.kernel_size_list)
            if number_of_candidates > 10:
                logger.warning("The number of number_of_candidates is : {}".format(
                    number_of_candidates))
            self.conv = nn.ModuleDict()
            for kernel in self.kernel_size_list:
                for channel in self.max_in_channels:
                    key_ = "{}_{}".format(kernel, channel)
                    self.conv[key_] = self.conv[str(kernel)] = nn.Conv2d(
                        channel, channel, kernel, self.stride,
                        groups=channel, bias=False,
                    )

    def get_active_filter(self, in_channel, kernel_size):
        """
        Only used when the weight sharing mode is 1, transform the kernel from large to small one
        """
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel,
                                   :in_channel, start:end, start:end]
        if self.weight_sharing_mode == 1 and kernel_size < max_kernel_size:
            # start with max kernel
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]
            for i in range(len(self.kernel_size_list) - 1, 0, -1):
                src_ks = self.kernel_size_list[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self.kernel_size_list[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__(
                        '%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(
                    0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel=None):
        _kernel = self.active_kernel if kernel is None else kernel
        _in_channel = x.size(1)
        if self.weight_sharing_mode == 0 or self.weight_sharing_mode == 1:
            filters = self.get_active_filter(_in_channel, _kernel).contiguous()
        elif self.weight_sharing_mode == 2:
            filters = self.conv[str(
                kernel)].weight[:_in_channel, :_in_channel, :, :]
        else:
            filters = self.conv["{}_{}".format(
                int(_kernel), int(_in_channel))].weight
        padding = get_same_padding(_kernel)
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, _in_channel
        )
        return y


class DynamicPointConv2d(nn.Module):
    def __init__(self, in_channel_list, out_channel_list, kernel_size=1, stride=1, dilation=1, weight_sharing=True):
        super(DynamicPointConv2d, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.max_in_channels = max(in_channel_list)
        self.max_out_channels = max(out_channel_list)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.weight_sharing = weight_sharing
        if self.weight_sharing:
            self.conv = nn.Conv2d(
                self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
            )
        else:
            self.conv = nn.ModuleDict()
            for _in_channel in self.in_channel_list:
                for _out_channel in self.out_channel_list:
                    self.conv["{}_{}".format(_in_channel, _out_channel)] = nn.Conv2d(
                        _in_channel, _out_channel, self.kernel_size, stride=self.stride, bias=False,
                    )

        self.active_out_channel = self.max_out_channels

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        if self.weight_sharing:
            filters = self.conv.weight[:out_channel,
                                       :in_channel, :, :].contiguous()
        else:
            filters = self.conv["{}_{}".format(
                int(in_channel), out_channel)].weight

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicLinear(nn.Module):
    def __init__(self, in_feature_list, out_feature_list, bias=True, weight_sharing=True):
        super(DynamicLinear, self).__init__()
        self.in_feature_list = in_feature_list
        self.out_feature_list = out_feature_list
        self.max_in_features = max(self.in_feature_list)
        self.max_out_features = max(self.out_feature_list)
        self.bias = bias
        self.weight_sharing = weight_sharing

        if self.weight_sharing:
            self.linear = nn.Linear(self.max_in_features,
                                    self.max_out_features, self.bias)
        else:
            self.linear = nn.ModuleDict()
            for _in_channel in self.in_feature_list:
                for _out_channel in self.out_feature_list:
                    self.linear["{}_{}".format(_in_channel, _out_channel)] = nn.Linear(self._in_channel,
                                                                                       self._out_channel, self.bias)
        self.active_out_features = self.max_out_features

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        if self.weight_sharing:
            weight = self.linear.weight[:out_features,
                                        :in_features].contiguous()
            bias = self.linear.bias[:out_features] if self.bias else None
        else:
            _name = "{}_{}".format(in_features, out_features)
            weight = self.linear[_name].weight
            bias = self.linear[_name].bias
        y = F.linear(x, weight, bias)
        return y


class DynamicSE(SEModule):

    def __init__(self, max_channel, reduction_list):
        super(DynamicSE, self).__init__(max_channel, max(reduction_list))
        self.reduction_list = reduction_list

    def forward(self, x, reduction=0.25):
        self.reduction = reduction
        in_channel = x.size(1)
        num_mid = make_divisible(int(in_channel * self.reduction), divisor=8)

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid,
                                           :in_channel, :, :].contiguous()
        reduce_bias = reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel,
                                           :num_mid, :, :].contiguous()
        expand_bias = expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y


class DynamicBatchNorm2d(nn.Module):
    # from https://github.com/mit-han-lab/once-for-all
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
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / \
                            float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x, bn.running_mean[:feature_dim], bn.running_var[:
                                                                 feature_dim], bn.weight[:feature_dim],
                bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


# dynamic layers

class DynamicMBConvLayer(nn.Module):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=None, expand_ratio_list=None, act_func_list=None,
                 se_list=None, stride=1, weight_sharing_mode=None):
        super(DynamicMBConvLayer, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = [
            3, 5, 7] if kernel_size_list is None else kernel_size_list
        self.expand_ratio_list = [
            3, 6] if expand_ratio_list is None else expand_ratio_list
        self.act_func_list = [
            'relu6', 'h_swish'] if act_func_list is None else act_func_list
        self.se_list = [0, 0.25] if se_list is None else se_list

        # build modules
        max_middle_channel = round(
            max(self.in_channel_list) * max(self.expand_ratio_list))
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicPointConv2d(
                    max(self.in_channel_list), max_middle_channel)),
                ('bn', DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))
