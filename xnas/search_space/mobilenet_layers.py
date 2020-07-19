from xnas.search_space.mobilenet_ops import *
from xnas.search_space.utils import adjust_bn_according_to_idx, copy_bn


class DynamicMBConvLayer(nn.Module):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=None, expand_ratio_list=None, act_func_list=None,
                 se_list=None, stride=1, weight_sharing_mode=0, weight_sharing_mode_conv=0):
        """
    DynamicMBConvLayer is a separable convolution operations with inverted_bottleneck, dynamic act, depth kernel size and input channels
    in_channel_list: type list example -> [24, 32, 68, xxx]
    kernel_size_list: type list example -> [3, 5, 7]
    expand_ratio_list: type list example -> [3, 6]
    act_list: type list example -> ['relu6', 'swish']
    weight_sharing_mode: type int example -> 0
        if weight_sharing_mode == 0:
            all the weight are from 1 big tensor e.g.68x408; The weight_sharing=True in DynamicPointConv2d
        if weight_sharing_mode == 1:
            we only share the weight with different kernels; The weight_sharing=False in DynamicPointConv2d
        if weight_sharing_mode == 2:
            we create different DynamicPointConv2d for different kernel sizes, and The weight_sharing=True in DynamicPointConv2d
        if weight_sharing_mode == 3:
            we create different DynamicPointConv2d for different kernel sizes, and The weight_sharing=True in DynamicPointConv2d
    weight_sharing_mode_conv: type int example -> 0：
        if weight_sharing_mode_conv == 0:
            all the weight are from 1 big tensor e.g. 408x7x7
        if weight_sharing_mode_conv == 1:
            all the weight are from 1 big tensor, e.g. 408x7x7, for different kernel size it will transform with matrix multi,
            which is identical with https://github.com/mit-han-lab/once-for-all
        if weight_sharing_mode_conv == 2:
            the weight of different kernels have different weight tensors, for example, if kernel_size_list=[3, 5, 7] and
            in_channel_list=[24, 32, 68], we have 3 weight tensors, 68x3x3, 68x5x5, 68x7x7
        if weight_sharing_mode_conv == 3:
            the weight of different kernels and different channels have different weight tensors
    """
        super(DynamicMBConvLayer, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = [
            3, 5, 7] if kernel_size_list is None else kernel_size_list
        self.in_channel_list.sort()
        self.expand_ratio_list = [
            3, 6] if expand_ratio_list is None else expand_ratio_list
        self.expand_ratio_list.sort()
        self.act_func_list = [
            'relu6', 'h_swish'] if act_func_list is None else act_func_list
        self.act_func_list.sort()
        self.se_list = [0, 0.25] if se_list is None else se_list
        self.se_list.sort()

        self.weight_sharing_mode = weight_sharing_mode
        self.weight_sharing_mode_conv = weight_sharing_mode_conv
        self.act = nn.ModuleDict()
        for act_name in self.act_func_list:
            self.act[act_name] = build_activation(act_name)

        # build modules
        middle_channel_list = [
            int(i*j) for j in expand_ratio_list for i in in_channel_list]
        # build depthsie convolution
        self.depth_conv = DynamicSeparableConv2d(
            middle_channel_list, self.kernel_size_list, weight_sharing_mode=weight_sharing_mode_conv, stride=stride, dilation=1)
        if self.weight_sharing_mode == 0 or self.weight_sharing_mode == 1:
            _weight_sharing = True if self.weight_sharing_mode == 0 else False
            self.inverted_bottleneck_conv = DynamicPointConv2d(
                self.in_channel_list, middle_channel_list, weight_sharing=_weight_sharing)
            self.inverted_bottleneck_bn = DynamicBatchNorm2d(
                middle_channel_list, weight_sharing=_weight_sharing)
            self.depth_se = DynamicSE(
                middle_channel_list, self.se_list, weight_sharing=_weight_sharing)
            self.depth_bn = DynamicBatchNorm2d(
                middle_channel_list, weight_sharing=_weight_sharing)
            self.point_linear_conv = DynamicPointConv2d(
                middle_channel_list, self.out_channel_list, weight_sharing=_weight_sharing)
            self.point_linear_bn = DynamicBatchNorm2d(
                self.out_channel_list, weight_sharing=_weight_sharing)
        elif self.weight_sharing_mode == 2 or self.weight_sharing_mode == 3:
            _weight_sharing = True if self.weight_sharing_mode == 2 else False
            for i, kernel in enumerate(self.kernel_size_list):
                if i == 0:
                    self.inverted_bottleneck_conv = nn.ModuleDict()
                    self.inverted_bottleneck_bn = nn.ModuleDict()
                    self.depth_se = nn.ModuleDict()
                    self.depth_bn = nn.ModuleDict()
                    self.point_linear_conv = nn.ModuleDict()
                    self.point_linear_bn = nn.ModuleDict()
                self.inverted_bottleneck_conv[str(kernel)] = DynamicPointConv2d(
                    self.in_channel_list, middle_channel_list, weight_sharing=_weight_sharing)
                self.inverted_bottleneck_bn[str(kernel)] = DynamicBatchNorm2d(
                    middle_channel_list, weight_sharing=_weight_sharing)
                self.depth_se[str(kernel)] = DynamicSE(
                    middle_channel_list, self.se_list, weight_sharing=_weight_sharing)
                self.depth_bn[str(kernel)] = DynamicBatchNorm2d(
                    middle_channel_list, weight_sharing=_weight_sharing)
                self.point_linear_conv[str(kernel)] = DynamicPointConv2d(
                    middle_channel_list, self.out_channel_list, weight_sharing=_weight_sharing)
                self.point_linear_bn[str(kernel)] = DynamicBatchNorm2d(
                    self.out_channel_list, weight_sharing=_weight_sharing)
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck_conv = None
            self.inverted_bottleneck_bn = None

    def sample_check(sample):
        assert sample is dict, "Sample shoud be a python dict!"
        assert 'expand' in sample.keys(), "expand should in sample"
        assert 'out_channel' in sample.keys(), "out_channel should in sample"
        assert 'act' in sample.keys(), "act should in sample"
        assert 'kernel' in sample.keys(), "kernel should in sample"
        assert 'se' in sample.keys(), "se should in sample"

    def forward(self, x, sample):
        self.sample_check(sample)
        in_channel = x.size(1)
        mid_channel = in_channel * sample['expand']
        out_channel = sample['out_channel']
        _act = self.act[sample['act']]

        if self.weight_sharing_mode == 0 or self.weight_sharing_mode == 1:
            _inverted_bottleneck_conv = self.inverted_bottleneck_conv
            _inverted_bottleneck_bn = self.inverted_bottleneck_bn
            _depth_se = self.depth_se
            _depth_bn = self.depth_bn
            _point_linear_conv = self.point_linear_conv
            _point_linear_bn = self.point_linear_bn
        elif self.weight_sharing_mode == 2 or self.weight_sharing_mode == 3:
            _inverted_bottleneck_conv = self.inverted_bottleneck_conv[str(
                sample['kernel'])]
            _inverted_bottleneck_bn = self.inverted_bottleneck_bn[str(
                sample['kernel'])]
            _depth_se = self.depth_se[str(sample['kernel'])]
            _depth_bn = self.depth_bn[str(sample['kernel'])]
            _point_linear_conv = self.point_linear_conv[str(sample['kernel'])]
            _point_linear_bn = self.point_linear_bn[str(sample['kernel'])]
        else:
            raise NotImplementedError

        # invert
        if self.inverted_bottleneck_conv is not None:
            x = _inverted_bottleneck_conv(x, mid_channel)
            x = _inverted_bottleneck_bn(x)
            x = _act(x)
        # depth wise conv
        x = self.depth_conv(x, sample['kernel'])
        x = _depth_bn(x)
        x = _act(x)
        x = _depth_se(x, sample['se'])
        # output
        x = _point_linear_conv(x, out_channel)
        x = _point_linear_bn(x)
        return x

    def get_active_subnet(self, in_channel, sample, preserve_weight=True):
        self.sample_check(sample)
        middle_channel = make_divisible(
            round(in_channel * sample['expand']), 8)

        # build the new layer
        sub_layer = MBInvertedConvLayer(
            in_channel, sample['out_channel'], sample['kernel'], self.stride, sample['expand'],
            act_func=sample['act'], mid_channels=middle_channel, use_se=self.use_se,
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[:middle_channel,
                                                               :in_channel, :, :]
            )
            copy_bn(sub_layer.inverted_bottleneck.bn,
                    self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(
                middle_channel, self.active_kernel_size).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(
                middle_channel // SEModule.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid,
                                                         :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.fc.reduce.bias.data[:se_mid])

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel,
                                                         :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.fc.expand.bias.data[:middle_channel])

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[:
                                                    self.active_out_channel, :middle_channel, :, :]
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = torch.sum(
            torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3))
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width = sorted_expand_list[expand_ratio_stage]
            target_width = round(max(self.in_channel_list) * target_width)
            importance[target_width:] = torch.arange(
                0, target_width - importance.size(0), -1)

        sorted_importance, sorted_idx = torch.sort(
            importance, dim=0, descending=True)
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
                se_expand.weight.data, 0, sorted_idx)
            se_expand.bias.data = torch.index_select(
                se_expand.bias.data, 0, sorted_idx)
            # se reduce: input dim 1 reorganize
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(
                se_reduce.weight.data, 1, sorted_idx)
            # middle weight reorganize
            se_importance = torch.sum(
                torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(
                se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(
                se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(
                se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(
                se_reduce.bias.data, 0, se_idx)

        # TODO if inverted_bottleneck is None, the previous layer should be reorganized accordingly
        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(
                self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx


class DynamicChannelConvLayer(MyModule):
    """
    Dynamic channel and activation function with fixed kernel size
    """

    def __init__(self, in_channel_list, out_channel_list, act_func_list=None, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, weight_sharing=True):
        super(DynamicChannelConvLayer, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.act_func_list = [
            'relu6'] if act_func_list is None else act_func_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.weight_sharing = weight_sharing

        self.conv = DynamicChannelConv2d(
            in_channel_list, out_channel_list, kernel_size=kernel_size, stride=stride, dilation=dilation, weight_sharing=weight_sharing)
        if use_bn:
            self.bn = DynamicBatchNorm2d(out_channel_list)
        self.act = nn.ModuleDict()
        for act_name in self.act_func_list:
            self.act[act_name] = build_activation(act_name)

    def forward(self, x, sample):

        x = self.conv(x, sample['out_channel'])
        if self.use_bn:
            x = self.bn(x)
        x = self.act[sample['act']](x)
        return x
