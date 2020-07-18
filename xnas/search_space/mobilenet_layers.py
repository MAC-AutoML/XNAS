from xnas.search_space.mobilenet_ops import *


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

    def forward(self, x, sample):
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
