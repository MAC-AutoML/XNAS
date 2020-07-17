# This code is highly referenced from onece-for-all from https://github.com/mit-han-lab/once-for-all
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

    def __init__(self, in_channel_list, kernel_size_list, act_list, weight_sharing_mode=0, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()
        assert weight_sharing_mode > 3, "The weight sharing mode should be less than 3"
        in_channel_list = in_channel_list.sort()
        kernel_size_list = kernel_size_list.sort()
        self.in_channel_list = in_channel_list
        self.kernel_size_list = kernel_size_list
        self.act_list = act_list
        self.max_in_channels = max(in_channel_list)
        self.max_kernel_size = max(kernel_size_list)
        self.stride = stride
        self.dilation = dilation
        self.weight_sharing_mode = weight_sharing_mode

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
        _kernel = self.max_kernel_size if kernel is None else kernel
        _in_channel = x.size(1)
        if self.weight_sharing_mode == 0 or self.weight_sharing_mode == 1:
            filters = self.get_active_filter(_in_channel, _kernel).contiguous()
        elif self.weight_sharing_mode == 2:
            filters = self.conv[str(
                kernel)].weight[:_in_channel, :_in_channel, :, :]
        else:
            filters = self.conv["{}_{}".format(_kernel, _in_channel)].weight
        padding = get_same_padding(_kernel)
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, _in_channel
        )
        return y


class DynamicPointConv2d(nn.Module):
    # from https://github.com/mit-han-lab/once-for-all
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicPointConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.conv.weight[:out_channel,
                                   :in_channel, :, :].contiguous()

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicLinear(nn.Module):
    # from https://github.com/mit-han-lab/once-for-all
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features,
                                self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.linear.weight[:out_features, :in_features].contiguous()
        bias = self.linear.bias[:out_features] if self.bias else None
        y = F.linear(x, weight, bias)
        return y


class DynamicSE(SEModule):

    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def forward(self, x, reduction=4):
        self.reduction = reduction
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction, divisor=8)

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
