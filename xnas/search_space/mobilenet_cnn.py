from xnas.search_space.mobilenet_layers import *
import itertools


def _to_list(input_):
    if input_ is list:
        return input_
    else:
        return [input_]


def get_search_operations(x, num_of_layers):
    if x[0] is list and len(x) == 1:
        return [x[0] for i in range(num_of_layers)]
    elif len(x) == num_of_layers:
        if x[0] is list:
            return x
        return [_to_list(i) for i in x]
    else:
        raise NotImplementedError


def make_divisible_list(_input, divsor):
    assert _input is list
    return list(dict.fromkeys([make_divisible(i, divisor) for i in _input]))


class MobileNetSearchSpace(nn.Module):
    def __init__(self, num_class, width_multi=1.0, stem_w=16, stem_act='swish', kernel_sizes=None,
                 strides=None, base_widths=None, expand_ratios=None,
                 se_ratios=None, acts=None, head_ws=None, head_acts=None,
                 weight_sharing_mode=0, weight_sharing_mode_conv=0):

        super(MobileNetSearchSpace, self).__init__()
        self.width_multi = width_multi if type(
            width_multi) is list else _to_list(width_multi)
        self.stem_w = make_divisible_list(stem_w) if type(stem_w) is list else _to_list(stem_w)
        self.stem_act = stem_act if type(stem_act) is list else _to_list(stem_act)
        # mobilenetV3 as basic search backbone
        self.strides = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1] if strides is None else strides
        self.base_widths = [16, 24, 24,  40,  40,  40, 80, 80, 80, 80, 112,
                            112, 160, 160, 160] if base_widths is None else base_widths
        self.kernel_sizes = [[3, 5, 7]] if kernel_sizes is None else kernel_sizes
        self.expand_ratios = [[3, 6]] if expand_ratios is None else expand_ratios
        self.se_ratios = [0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4] if se_ratios is None else se_ratios
        self.acts = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'swish', 'swish', 'swish',
                     'swish', 'swish', 'swish', 'swish', 'swish', 'swish'] if acts is None else acts
        self.head_ws = [960, 1280] if head_ws is None else head_ws
        self.head_acts = ['swish', 'swish'] if head_acts is None else head_acts
        _weight_sharing = True if weight_sharing_mode == 0 or weight_sharing_mode == 2 else False

        # build stem in convolutional operations
        self.stem_layer = DynamicChannelConvLayer(
            [3], self.stem_w, act_func_list=self.stem_act, kernel_size=3, stride=2, dilation=1, use_bn=True, weight_sharing=_weight_sharing)
        # build main blocks
        _number_of_layers = len(self.strides)
        self.kernel_sizes = get_search_operations(self.kernel_sizes, _number_of_layers)
        self.expand_ratios = get_search_operations(self.expand_ratios, _number_of_layers)
        self.acts = get_search_operations(self.acts, _number_of_layers)
        self.se_ratios = get_search_operations(self.se_ratios, _number_of_layers)
        self.out_channel_lists = [list(dict.fromkeys([make_divisible(i*j, 8)
                                                      for j in self.width_multi])) for i in self.base_widths]
        self.in_channel_list = self.stem_w + [list(dict.fromkeys([make_divisible(i*j, 8)
                                                                  for j in self.width_multi])) for i in self.base_widths[:-1]]
        # add
        self.MBblocks = nn.ModuleList()
        for (_kernel, _expand_ratio, _act_func, _in_channel_list, _out_channel_list, _se, _stride) in zip(self.kernel_sizes, self.expand_ratios, self.acts, self.in_channel_list, self.out_channel_lists, self.se_ratios, self.strides):
            self.MBblocks.append(DynamicMBConvLayer(_in_channel_list, _out_channel_list,
                                                    kernel_size_list=_kernel, expand_ratio_list=_expand_ratio, act_func_list=_act_func,
                                                    se_list=_se, stride=_stride, weight_sharing_mode=weight_sharing_mode, weight_sharing_mode_conv=weight_sharing_mode_conv))
        # build mobilenet heads
        self.head_ws = get_search_operations(self.head_ws)
        self.head_acts = get_search_operations(self.head_acts)
        self.MBHead = nn.ModuleList()
        for i, _output_channel in enumerate(self.head_ws):
            if i == 0:
                self.MBHead.append(DynamicChannelConvLayer(
                    self.out_channel_lists[-1], _output_channel, act_func_list=self.head_acts[i], kernel_size=1, stride=1, dilation=1, use_bn=True, weight_sharing=_weight_sharing))
            else:
                self.MBHead.append(DynamicLinearLayer(
                    self.head_ws[i-1], self.head_ws[i], act_func_list=self.head_acts[i], weight_sharing=_weight_sharing, bias=True, dropout_rate=0))
        self.fc = DynamicLinear(self.head_ws[-1], [num_class], bias=True, weight_sharing=_weight_sharing)
        self.num_edge_ops()

    def num_edge_ops(self):
        _num_edges = 0
        _num_ops = []
        self.search_able = {}
        if len(self.stem_layer._sample_operations) > 1:
            self.search_able['stem'] = True
            _num_edges += 1
            _num_ops.append(len(self.stem_layer._sample_operations))
        else:
            self.search_able['stem'] = False
        self.search_able['blocks'] = []
        for block in self.MBblocks:
            if len(block._sample_operations) > 1:
                _num_edges += 1
                _num_ops.append(len(self.stem_layer._sample_operations))
                self.search_able['blocks'].append(True)
            else:
                self.search_able['blocks'].append(False)
        self.search_able['head'] = []
        for block in self.MBHead:
            if len(block._sample_operations) > 1:
                _num_edges += 1
                _num_ops.append(len(self.stem_layer._sample_operations))
                self.search_able['head'].append(True)
            else:
                self.search_able['head'].append(False)
        self.num_edges = _num_edges
        self.num_ops = _num_ops

    def forward(self, x, sample):
        pass
