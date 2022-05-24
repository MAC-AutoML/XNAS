import torch.nn as nn

from xnas.spaces.OFA.utils import make_divisible
from xnas.spaces.OFA.ops import (
    set_layer_from_config,
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    ResidualBlock,
    ResNetBottleneckBlock,
    GlobalAvgPool2d
)
from xnas.spaces.OFA.MobileNetV3.cnn import WSConv_Network


__all__ = ["ResNet", "ResNet50", "ResNet50D"]


class ResNet(WSConv_Network):

    BASE_DEPTH_LIST = [2, 2, 4, 2]
    STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

    def __init__(self, input_stem, blocks, classifier):
        super(ResNet, self).__init__()

        self.input_stem = nn.ModuleList(input_stem)
        self.max_pooling = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pool = GlobalAvgPool2d(keep_dim=False)
        self.classifier = classifier

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        x = self.max_pooling(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            _str += layer.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": ResNet.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        classifier = set_layer_from_config(config["classifier"])

        input_stem = []
        for layer_config in config["input_stem"]:
            input_stem.append(set_layer_from_config(layer_config))
        blocks = []
        for block_config in config["blocks"]:
            blocks.append(set_layer_from_config(block_config))

        net = ResNet(input_stem, blocks, classifier)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResNetBottleneckBlock) and isinstance(
                m.downsample, IdentityLayer
            ):
                m.conv3.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks):
            if (
                not isinstance(block.downsample, IdentityLayer)
                and len(block_index_list) > 0
            ):
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def load_state_dict(self, state_dict, **kwargs):
        super(ResNet, self).load_state_dict(state_dict)


class ResNet50(ResNet):
    def __init__(
        self,
        n_classes=1000,
        width_mult=1.0,
        bn_param=(0.1, 1e-5),
        dropout_rate=0,
        expand_ratio=None,
        depth_param=None,
    ):

        expand_ratio = 0.25 if expand_ratio is None else expand_ratio

        input_channel = make_divisible(64 * width_mult)
        stage_width_list = ResNet.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(width * width_mult)

        depth_list = [3, 4, 6, 3]
        if depth_param is not None:
            for i, depth in enumerate(ResNet.BASE_DEPTH_LIST):
                depth_list[i] = depth + depth_param

        stride_list = [1, 2, 2, 2]

        # build input stem
        input_stem = [
            ConvLayer(
                3,
                input_channel,
                kernel_size=7,
                stride=2,
                use_bn=True,
                act_func="relu",
                ops_order="weight_bn_act",
            )
        ]

        # blocks
        blocks = []
        for d, width, s in zip(depth_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = ResNetBottleneckBlock(
                    input_channel,
                    width,
                    kernel_size=3,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    act_func="relu",
                    downsample_mode="conv",
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(ResNet50, self).__init__(input_stem, blocks, classifier)

        # set bn param
        self.set_bn_param(*bn_param)


class ResNet50D(ResNet):
    def __init__(
        self,
        n_classes=1000,
        width_mult=1.0,
        bn_param=(0.1, 1e-5),
        dropout_rate=0,
        expand_ratio=None,
        depth_param=None,
    ):

        expand_ratio = 0.25 if expand_ratio is None else expand_ratio

        input_channel = make_divisible(64 * width_mult)
        mid_input_channel = make_divisible(input_channel // 2)
        stage_width_list = ResNet.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(width * width_mult)

        depth_list = [3, 4, 6, 3]
        if depth_param is not None:
            for i, depth in enumerate(ResNet.BASE_DEPTH_LIST):
                depth_list[i] = depth + depth_param

        stride_list = [1, 2, 2, 2]

        # build input stem
        input_stem = [
            ConvLayer(3, mid_input_channel, 3, stride=2, use_bn=True, act_func="relu"),
            ResidualBlock(
                ConvLayer(
                    mid_input_channel,
                    mid_input_channel,
                    3,
                    stride=1,
                    use_bn=True,
                    act_func="relu",
                ),
                IdentityLayer(mid_input_channel, mid_input_channel),
            ),
            ConvLayer(
                mid_input_channel,
                input_channel,
                3,
                stride=1,
                use_bn=True,
                act_func="relu",
            ),
        ]

        # blocks
        blocks = []
        for d, width, s in zip(depth_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = ResNetBottleneckBlock(
                    input_channel,
                    width,
                    kernel_size=3,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    act_func="relu",
                    downsample_mode="avgpool_conv",
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(ResNet50D, self).__init__(input_stem, blocks, classifier)

        # set bn param
        self.set_bn_param(*bn_param)
