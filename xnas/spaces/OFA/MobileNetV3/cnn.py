from collections.abc import Iterable
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from xnas.spaces.OFA.ops import *
from xnas.spaces.OFA.utils import min_divisible_value


__all__ = ["WSConv_Network", "MobileNetV3", "MobileNetV3Large"]


class WSConv_Network(nn.Module):
    """Network with all Conv2d replaced by Weight Standard Conv2d."""

    def set_bn_param(self, momentum, eps, gn_channel_per_group=None, ws_eps=None, **kwargs):
    
        """Replace BN with GN"""
        if gn_channel_per_group is None:
            return

        for m in self.modules():
            to_replace_dict = {}
            for name, sub_m in m.named_children():
                if isinstance(sub_m, nn.BatchNorm2d):
                    num_groups = sub_m.num_features // min_divisible_value(
                        sub_m.num_features, gn_channel_per_group
                    )
                    gn_m = nn.GroupNorm(
                        num_groups=num_groups,
                        num_channels=sub_m.num_features,
                        eps=sub_m.eps,
                        affine=True,
                    )

                    # load weight
                    gn_m.weight.data.copy_(sub_m.weight.data)
                    gn_m.bias.data.copy_(sub_m.bias.data)
                    # load requires_grad
                    gn_m.weight.requires_grad = sub_m.weight.requires_grad
                    gn_m.bias.requires_grad = sub_m.bias.requires_grad

                    to_replace_dict[name] = gn_m
            m._modules.update(to_replace_dict)

        """Init Norm params"""
        for m in self.modules():
            if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
                m.momentum = momentum
                m.eps = eps
            elif isinstance(m, nn.GroupNorm):
                m.eps = eps

        """Replace Conv2d with WeightStandardConv2d"""
        if ws_eps is None:
            return

        for m in self.modules():
            to_update_dict = {}
            for name, sub_module in m.named_children():
                if isinstance(sub_module, nn.Conv2d) and not sub_module.bias:
                    # only replace conv2d layers that are followed by normalization layers (i.e., no bias)
                    to_update_dict[name] = sub_module
            for name, sub_module in to_update_dict.items():
                m._modules[name] = WeightStandardConv2d(
                    sub_module.in_channels,
                    sub_module.out_channels,
                    sub_module.kernel_size,
                    sub_module.stride,
                    sub_module.padding,
                    sub_module.dilation,
                    sub_module.groups,
                    sub_module.bias,
                )
                # load weight
                m._modules[name].load_state_dict(sub_module.state_dict())
                # load requires_grad
                m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
                if sub_module.bias is not None:
                    m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
        # set ws_eps
        for m in self.modules():
            if isinstance(m, WeightStandardConv2d):
                m.WS_EPS = ws_eps

    def get_bn_param(self):
        ws_eps = None
        for m in self.modules():
            if isinstance(m, WeightStandardConv2d):
                ws_eps = m.WS_EPS
                break
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    "momentum": m.momentum,
                    "eps": m.eps,
                    "ws_eps": ws_eps,
                }
            elif isinstance(m, nn.GroupNorm):
                return {
                    "momentum": None,
                    "eps": m.eps,
                    "gn_channel_per_group": m.num_channels // m.num_groups,
                    "ws_eps": ws_eps,
                }
        return None

    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)


class MobileNetV3(WSConv_Network):
    def __init__(
        self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
    ):
        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.global_avg_pool = GlobalAvgPool2d(keep_dim=True)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = self.global_avg_pool(x)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        _str += self.final_expand_layer.module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": MobileNetV3.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "final_expand_layer": self.final_expand_layer.config,
            "feature_mix_layer": self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config["first_conv"])
        final_expand_layer = set_layer_from_config(config["final_expand_layer"])
        feature_mix_layer = set_layer_from_config(config["feature_mix_layer"])
        classifier = set_layer_from_config(config["classifier"])

        blocks = []
        for block_config in config["blocks"]:
            blocks.append(ResidualBlock.build_from_config(block_config))

        net = MobileNetV3(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

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

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3,
            input_channel,
            kernel_size=3,
            stride=2,
            use_bn=True,
            act_func="h_swish",
            ops_order="weight_bn_act",
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for (
                k,
                mid_channel,
                out_channel,
                use_se,
                act_func,
                stride,
                expand_ratio,
            ) in block_config_list:
                mb_conv = MBConvLayer(
                    feature_dim,
                    out_channel,
                    int(k),
                    stride,
                    expand_ratio,
                    mid_channel,
                    act_func,
                    use_se,
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim,
            feature_dim * 6,
            kernel_size=1,
            use_bn=True,
            act_func="h_swish",
            ops_order="weight_bn_act",
        )
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim * 6,
            last_channel,
            kernel_size=1,
            bias=False,
            use_bn=False,
            act_func="h_swish",
        )
        # classifier
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    @staticmethod
    def adjust_cfg(
        cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None
    ):
        if depth_param is not None and not isinstance(depth_param, Iterable):
            depth_param = [depth_param] * 5
            total_depth = sum(depth_param)
        if ks is not None and not isinstance(ks, Iterable):
            ks = [ks] * total_depth
        if expand_ratio is not None and not isinstance(expand_ratio, Iterable):
            expand_ratio = [expand_ratio] * total_depth
        depth_cnt = 0
        for i, (stage_id, block_config_list) in enumerate(sorted(cfg.items(), key=lambda x:int(x[0]))):
            if depth_param is not None and stage_id != "0":
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [
                    deepcopy(block_config_list[-1]) for _ in range(depth_param[i-1] - 1)
                ]
                cfg[stage_id] = new_block_config_list
            for block_config in cfg[stage_id]:
                if stage_id != "0":
                    if ks is not None:
                        block_config[0] = ks[depth_cnt]
                    if expand_ratio is not None:
                        block_config[-1] = expand_ratio[depth_cnt]
                        block_config[1] = None
                        if stage_width_list is not None:
                            block_config[2] = stage_width_list[i]
                    depth_cnt += 1
        return cfg

    def load_state_dict(self, state_dict, **kwargs):
        current_state_dict = self.state_dict()

        for key in state_dict:
            if key not in current_state_dict:
                assert ".mobile_inverted_conv." in key
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            current_state_dict[new_key] = state_dict[key]
        super(MobileNetV3, self).load_state_dict(current_state_dict)


class MobileNetV3Large(MobileNetV3):
    def __init__(
        self,
        n_classes=1000,
        width_mult=1.0,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.2,
        ks=None, # chosen from {3, 5, 7}
        expand_ratio=None, # chosen from {3,4,6}
        depth_param=None, # chosen from {2,3,4}
        stage_width_list=None,
    ):
        input_channel = 16
        last_channel = 1280
        
        if depth_param is not None and not isinstance(depth_param, Iterable):
            depth_param = [depth_param] * 5
            total_depth = sum(depth_param)
        if ks is not None and not isinstance(ks, Iterable):
            ks = [ks] * total_depth
        if expand_ratio is not None and not isinstance(expand_ratio, Iterable):
            expand_ratio = [expand_ratio] * total_depth

        input_channel = make_divisible(input_channel * width_mult)
        last_channel = (
            make_divisible(last_channel * width_mult)
            if width_mult > 1.0
            else last_channel
        )

        cfg = {
            #    k,     exp,    c,      se,         nl,         s,      e,
            "0": [
                [3, 16, 16, False, "relu", 1, 1],
            ],
            "1": [
                [3, 64, 24, False, "relu", 2, None],  # 4
                [3, 72, 24, False, "relu", 1, None],  # 3
            ],
            "2": [
                [5, 72, 40, True, "relu", 2, None],  # 3
                [5, 120, 40, True, "relu", 1, None],  # 3
                [5, 120, 40, True, "relu", 1, None],  # 3
            ],
            "3": [
                [3, 240, 80, False, "h_swish", 2, None],  # 6
                [3, 200, 80, False, "h_swish", 1, None],  # 2.5
                [3, 184, 80, False, "h_swish", 1, None],  # 2.3
                [3, 184, 80, False, "h_swish", 1, None],  # 2.3
            ],
            "4": [
                [3, 480, 112, True, "h_swish", 1, None],  # 6
                [3, 672, 112, True, "h_swish", 1, None],  # 6
            ],
            "5": [
                [5, 672, 160, True, "h_swish", 2, None],  # 6
                [5, 960, 160, True, "h_swish", 1, None],  # 6
                [5, 960, 160, True, "h_swish", 1, None],  # 6
            ],
        }

        cfg = self.adjust_cfg(cfg, ks, expand_ratio, depth_param, stage_width_list)
        depth_param = [len(_[1]) for _ in sorted(cfg.items(), key=lambda x:int(x[0]))][1:]
        assert len(depth_param) == 5
        self.feature_idx = np.cumsum(depth_param)[[1, 3]]+1 # + stage-0
        # width multiplier on mobile setting, change `exp: 1` and `c: 2`
        for stage_id, block_config_list in cfg.items():
            for block_config in block_config_list:
                if block_config[1] is not None:
                    block_config[1] = make_divisible(block_config[1] * width_mult)
                block_config[2] = make_divisible(block_config[2] * width_mult)

        (
            first_conv,
            blocks,
            final_expand_layer,
            feature_mix_layer,
            classifier,
        ) = self.build_net_via_cfg(
            cfg, input_channel, last_channel, n_classes, dropout_rate
        )
        super(MobileNetV3Large, self).__init__(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )
        # set bn param
        self.set_bn_param(*bn_param)

    def forward_with_features(self, x, *args, **kwargs):
        x = self.first_conv(x)
        features = []
        for i, block in enumerate(self.blocks):
            if i in (self.feature_idx):
                features.append(x)
            x = block(x)
        x = self.final_expand_layer(x)
        features.append(x)
        assert len(features) == 3
        x = self.global_avg_pool(x)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return features, logits
    
def _MobileNetV3(*args, **kwargs):
    return MobileNetV3Large(*args, **kwargs)
