from model.mb_ops import *
from model.proxyless_cnn import ProxylessNASNets
from utils import flops_counter


class MobileNetV3(MyNetwork):

    def __init__(self, n_classes=1000, base_stage_width=None,
                 width_mult=1.2, conv_candidates=None, depth=4):
        super(MobileNetV3, self).__init__()

        self.width_mult = width_mult
        self.depth = depth
        self.base_stage_width = base_stage_width
        self.conv_candidates = [
                '3x3_MBConv3', '3x3_MBConv6',
                '5x5_MBConv3', '5x5_MBConv6',
                '7x7_MBConv3', '7x7_MBConv6',
            ] if conv_candidates is None else conv_candidates

        if self.base_stage_width == 'ofa':
            base_stage_width = [16, 24, 40, 80, 112, 160, 960, 1280]
        else:
            raise NotImplementedError
        self.base_stage_width = base_stage_width
        final_expand_width = make_divisible(base_stage_width[-2] * self.width_mult, 8)
        last_channel = make_divisible(base_stage_width[-1] * self.width_mult, 8)

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + [self.depth] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = make_divisible(base_width * self.width_mult, 8)
            width_list.append(width)

        input_channel = width_list[0]
        # first conv layer
        first_conv = ConvLayer(3, input_channel, kernel_size=3, stride=2, act_func='h_swish')
        first_block_conv = MBInvertedConvLayer(
            in_channels=input_channel, out_channels=input_channel, kernel_size=3, stride=stride_stages[0],
            expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0],
        )
        first_block = MobileInvertedResidualBlock(first_block_conv, IdentityLayer(input_channel, input_channel))

        # inverted residual blocks
        blocks = nn.ModuleList()
        blocks.append(first_block)
        feature_dim = input_channel
        self.candidate_ops = []

        for width, n_block, s, act_func, use_se in zip(width_list[1:], n_block_list[1:],
                                                       stride_stages[1:], act_stages[1:], se_stages[1:]):

            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                    # conv
                if stride == 1 and feature_dim == width:
                    modified_conv_candidates = conv_candidates + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates + ['3x3_MBConv1']
                self.candidate_ops.append(modified_conv_candidates)
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, feature_dim, width, stride, 'weight_bn_act',
                act_func=act_func, use_se=use_se), )
                if stride == 1 and feature_dim == width:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(conv_op, shortcut))
                feature_dim = width
        # final expand layer, feature mix layer & classifier
        final_expand_layer = ConvLayer(feature_dim, final_expand_width, kernel_size=1, act_func='h_swish')
        feature_mix_layer = ConvLayer(
            final_expand_width, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
        )
        classifier = LinearLayer(last_channel, n_classes)

        self.first_conv = first_conv
        self.blocks = blocks
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAMobileNetV3'

    def forward(self, x, sample):
        # first conv
        x = self.first_conv(x)
        assert len(self.blocks) - 1 == len(sample)
        for i in range(len(self.blocks[1:])):
            this_block_conv = self.blocks[i+1].mobile_inverted_conv
            if isinstance(this_block_conv, MixedEdge):
                this_block_conv.active_index = [sample[i]]
            else:
                raise NotImplementedError
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = self.global_avg_pooling(x)
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    def flops_counter_per_layer(self, input_size=None):
        self.eval()
        if input_size is None:
            input_size = [1, 3, 224, 224]
        original_device = self.parameters().__next__().device
        x = torch.zeros(input_size).to(original_device)
        first_conv_flpos, _ = flops_counter.profile(self.first_conv, input_size)
        x = self.first_conv(x)
        block_flops = []
        for block in self.blocks:
            if not isinstance(block.mobile_inverted_conv, MixedEdge):
                _flops, _ = flops_counter.profile(block, x.size())
                block_flops.append([_flops])
                x = block(x)
            else:
                _flops_list = []
                for i in range(block.mobile_inverted_conv.n_choices):
                    if isinstance(block.mobile_inverted_conv.candidate_ops[i], ZeroLayer):
                        _flops_list.append(0)
                    else:
                        _flops, _ = flops_counter.profile(block.mobile_inverted_conv.candidate_ops[i], x.size())
                        _flops_list.append(_flops)
                block_flops.append(_flops_list)
                x = block(x)
        final_expand_layer_flops, _ = flops_counter.profile(self.final_expand_layer, x.size())
        x = self.final_expand_layer(x)

        x = self.global_avg_pooling(x)
        feature_mix_layer_flops, _ = flops_counter.profile(self.feature_mix_layer, x.size())
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)  # flatten
        classifier_flops, _ = flops_counter.profile(self.classifier, x.size())
        self.train()
        return {'first_conv_flpos': first_conv_flpos,
                'block_flops': block_flops,
                'final_expand_layer_flops': final_expand_layer_flops,
                'feature_mix_layer_flops': feature_mix_layer_flops,
                'classifier_flops': classifier_flops}

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def genotype(self, theta):
        genotype = []
        for i in range(theta.shape[0]):
            genotype.append(self.candidate_ops[i][np.argmax(theta[i])])
        return genotype


def get_super_net(n_classes=1000, base_stage_width=None, width_mult=1.2, conv_candidates=None, depth=4):
    # proxyless, google,
    if base_stage_width in ['proxyless', 'google']:
        return ProxylessNASNets(n_classes=n_classes, base_stage_width=base_stage_width,
                                width_mult=width_mult, conv_candidates=conv_candidates,
                                depth=depth)
    elif base_stage_width == 'ofa':
        return MobileNetV3(n_classes=n_classes, base_stage_width=base_stage_width,
                           width_mult=width_mult, conv_candidates=conv_candidates,
                           depth=depth)
    else:
        raise NotImplementedError



