from xnas.search_space.mobilenet_layers import *


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
    def __init__(width_multi=1.0, stem_w=16, stem_act='swish', kernel_sizes=None,
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

        # build stem in convolutional operations
        self.stem_layer = DynamicChannelConvLayer(
            [3], self.stem_w, act_func_list=self.stem_act, kernel_size=3, stride=2, dilation=1, use_bn=True, weight_sharing=True)
        self.stem_sample = {'out_channel': None, 'act': None}
        # build blocks
        _number_of_layers = len(self.strides)
        self.kernel_sizes = get_search_operations(self.kernel_sizes, _number_of_layers)
        self.expand_ratios = get_search_operations(self.expand_ratios, _number_of_layers)
        self.out_channel_lists = [list(dict.fromkeys([make_divisible(i*j, 8)
                                                      for j in self.width_multi])) for i in self.base_widths]
        self.in_channel_list = self.stem_w + [list(dict.fromkeys([make_divisible(i*j, 8)
                                                                  for j in self.width_multi])) for i in self.base_widths[:-1]]
