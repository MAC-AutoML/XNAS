import json
import torch.nn as nn

from xnas.spaces.OFA.ops import (
    set_layer_from_config,
    MBConvLayer,
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    ResidualBlock,
    GlobalAvgPool2d,
)
from xnas.spaces.OFA.utils import val2list, make_divisible
from xnas.spaces.OFA.MobileNetV3.cnn import WSConv_Network


__all__ = ["proxyless_base", "ProxylessNASNet", "MobileNetV2"]


def proxyless_base(
    net_config=None,
    n_classes=None,
    bn_param=None,
    dropout_rate=None,
):
    assert net_config is not None, "Please input a network config"
    net_config_json = json.load(open(net_config, "r"))

    if n_classes is not None:
        net_config_json["classifier"]["out_features"] = n_classes
    if dropout_rate is not None:
        net_config_json["classifier"]["dropout_rate"] = dropout_rate

    net = ProxylessNASNet.build_from_config(net_config_json)
    if bn_param is not None:
        net.set_bn_param(*bn_param)

    return net


class ProxylessNASNet(WSConv_Network):
    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNet, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pool = GlobalAvgPool2d(keep_dim=False)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        _str += self.feature_mix_layer.module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": ProxylessNASNet.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "feature_mix_layer": None
            if self.feature_mix_layer is None
            else self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config["first_conv"])
        feature_mix_layer = set_layer_from_config(config["feature_mix_layer"])
        classifier = set_layer_from_config(config["classifier"])

        blocks = []
        for block_config in config["blocks"]:
            blocks.append(ResidualBlock.build_from_config(block_config))

        net = ProxylessNASNet(first_conv, blocks, feature_mix_layer, classifier)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if isinstance(m.conv, MBConvLayer) and isinstance(
                    m.shortcut, IdentityLayer
                ):
                    m.conv.point_linear.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks[1:], 1):
            if block.shortcut is None and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def load_state_dict(self, state_dict, **kwargs):
        current_state_dict = self.state_dict()

        for key in state_dict:
            if key not in current_state_dict:
                assert ".mobile_inverted_conv." in key
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            current_state_dict[new_key] = state_dict[key]
        super(ProxylessNASNet, self).load_state_dict(current_state_dict)


class MobileNetV2(ProxylessNASNet):
    def __init__(
        self,
        n_classes=1000,
        width_mult=1.0,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.2,
        ks=None, # a list only include {3, 5, 7}
        expand_ratio=None, # in proxyless space only 3 or 6
        depth_param=None,
        stage_width_list=None,
    ):

        ks = 3 if ks is None else ks
        expand_ratio = [6]*6 if expand_ratio is None else expand_ratio

        input_channel = 32
        last_channel = 1280

        input_channel = make_divisible(input_channel * width_mult)
        last_channel = (
            make_divisible(last_channel * width_mult)
            if width_mult > 1.0
            else last_channel
        )

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [None, 24, 2, 2],
            [None, 32, 3, 2],
            [None, 64, 4, 2],
            [None, 96, 3, 1],
            [None, 160, 3, 2],
            [None, 320, 1, 1],
        ]

        if depth_param is not None:
            assert isinstance(depth_param, int)
            for i in range(1, len(inverted_residual_setting) - 1):
                inverted_residual_setting[i][2] = depth_param

        if stage_width_list is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][1] = stage_width_list[i]

        if expand_ratio is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][0] = expand_ratio[i]

        ks = val2list(ks, sum([n for _, _, n, _ in inverted_residual_setting]) - 1)
        _pt = 0

        # first conv layer
        first_conv = ConvLayer(
            3,
            input_channel,
            kernel_size=3,
            stride=2,
            use_bn=True,
            act_func="relu6",
            ops_order="weight_bn_act",
        )
        # inverted residual blocks
        blocks = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                if t == 1: # only used for first block
                    kernel_size = 3
                else:
                    kernel_size = ks[_pt]
                    _pt += 1
                mobile_inverted_conv = MBConvLayer(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    expand_ratio=t,
                )
                if stride == 1:
                    if input_channel == output_channel:
                        shortcut = IdentityLayer(input_channel, input_channel)
                    else:
                        shortcut = None
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                input_channel = output_channel
        # 1x1_conv before global average pooling
        feature_mix_layer = ConvLayer(
            input_channel,
            last_channel,
            kernel_size=1,
            use_bn=True,
            act_func="relu6",
            ops_order="weight_bn_act",
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(MobileNetV2, self).__init__(
            first_conv, blocks, feature_mix_layer, classifier
        )

        # set bn param
        self.set_bn_param(*bn_param)
