from model.mb_layers import *
import pdb


def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride, ops_order,
                        act_func='relu6', use_se=False):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        'Identity': lambda in_C, out_C, S: IdentityLayer(in_C, out_C, ops_order=ops_order),
        'Zero': lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    # add MBConv layers
    name2ops.update({
        '3x3_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1, None, act_func, use_se),
        '3x3_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 2, None, act_func, use_se),
        '3x3_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3, None, act_func, use_se),
        '3x3_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 4, None, act_func, use_se),
        '3x3_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 5, None, act_func, use_se),
        '3x3_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6, None, act_func, use_se),
        #######################################################################################
        '5x5_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1, None, act_func, use_se),
        '5x5_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 2, None, act_func, use_se),
        '5x5_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3, None, act_func, use_se),
        '5x5_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 4, None, act_func, use_se),
        '5x5_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 5, None, act_func, use_se),
        '5x5_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6, None, act_func, use_se),
        #######################################################################################
        '7x7_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1, None, act_func, use_se),
        '7x7_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 2, None, act_func, use_se),
        '7x7_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3, None, act_func, use_se),
        '7x7_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 4, None, act_func, use_se),
        '7x7_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 5, None, act_func, use_se),
        '7x7_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6, None, act_func, use_se),
    })

    return [
        name2ops[name](in_channels, out_channels, stride) for name in candidate_ops
    ]


class MobileInvertedResidualBlock(MyNetwork):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class MixedEdge(MyModule):
    MODE = None  # full, two, None, full_v2

    def __init__(self, candidate_ops):
        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)

        self.active_index = [0]
        self.inactive_index = None

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    @property
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def is_zero_layer(self):
        return self.active_op.is_zero_layer()

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    """ """

    def forward(self, x):
        # output = 0
        # for i in self.active_index:
        #     oi = self.candidate_ops[i](x)
        #     output = output + oi
        # only support 1 selection
        assert len(self.active_index) == 1
        x = self.candidate_ops[self.active_index[0]](x)
        return x

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'Mix(%s, %.3f)' % (self.candidate_ops[chosen_index].module_str, probs)

    @staticmethod
    def name():
        return 'MixedEdge'

    @property
    def config(self):
        return {
            'name': MixedEdge.__name__,
            'selection': [i.config for i in self.candidate_ops],
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        """ Only active paths taken into consideration when calculating FLOPs """
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)


