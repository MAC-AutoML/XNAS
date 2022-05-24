"""
    code from https://github.com/megvii-model/SinglePathOneShot/
"""
    
import torch
import torch.nn as nn
import numpy as np


CHANNELS = [16,
           64, 64, 64, 64,
           160, 160, 160, 160,
           320, 320, 320, 320, 320, 320, 320, 320,
           640, 640, 640, 640]
LAST_CHANNEL = 1024


def channel_shuffle_spos(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class Choice_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, supernet=True):
        super(Choice_Block, self).__init__()
        padding = kernel // 2
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels

        self.cb_main = nn.Sequential(
            # pw
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=kernel, stride=stride, padding=padding,
                      bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw_linear
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel, stride=2, padding=padding,
                          bias=False, groups=self.in_channels),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                # pw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle_spos(x)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y


class Choice_Block_x(nn.Module):
    def __init__(self, in_channels, out_channels, stride, supernet=True):
        super(Choice_Block_x, self).__init__()
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels

        self.cb_main = nn.Sequential(
            # dw
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False, groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2,
                          padding=1, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                # pw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle_spos(x)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y


class SPOS_supernet(nn.Module):
    def __init__(self, dataset, resize, classes, layers):
        super(SPOS_supernet, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.classes = classes
        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, CHANNELS[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS[0], affine=False),
            nn.ReLU6(inplace=True)
        )
        # choice_block
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = CHANNELS[i], CHANNELS[i + 1]
            else:
                stride = 1
                inp, oup = CHANNELS[i] // 2, CHANNELS[i + 1]
            layer_cb = nn.ModuleList([])
            for j in self.kernel_list:
                if j == 'x':
                    layer_cb.append(Choice_Block_x(inp, oup, stride=stride))
                else:
                    layer_cb.append(Choice_Block(inp, oup, kernel=j, stride=stride))
            self.choice_block.append(layer_cb)
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(CHANNELS[-1], LAST_CHANNEL, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(LAST_CHANNEL, affine=False),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(LAST_CHANNEL, self.classes, bias=False)
        self._initialize_weights()
    
    def weights(self):
        return self.parameters()

    def forward(self, x, choice=np.random.randint(4, size=20)):
        x = self.stem(x)
        # repeat
        for i, j in enumerate(choice):
            x = self.choice_block[i][j](x)
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, LAST_CHANNEL)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class infer_SPOS(nn.Module):
    def __init__(self, dataset, resize, classes, layers, choice):
        super(infer_SPOS, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.classes = classes
        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, CHANNELS[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS[0]),
            nn.ReLU6(inplace=True)
        )
        # choice_block
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = CHANNELS[i], CHANNELS[i + 1]
            else:
                stride = 1
                inp, oup = CHANNELS[i] // 2, CHANNELS[i + 1]
            if choice[i] == 3:
                self.choice_block.append(Choice_Block_x(inp, oup, stride=stride, supernet=False))
            else:
                self.choice_block.append(
                    Choice_Block(inp, oup, kernel=self.kernel_list[choice[i]], stride=stride, supernet=False))
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(CHANNELS[-1], LAST_CHANNEL, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(LAST_CHANNEL),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(LAST_CHANNEL, self.classes, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        # repeat
        for i in range(self.layers):
            x = self.choice_block[i](x)
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, LAST_CHANNEL)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# build API
def _SPOS_CNN():
    from xnas.core.config import cfg
    return SPOS_supernet(
        dataset=cfg.LOADER.DATASET,
        resize=cfg.SPOS.RESIZE,
        classes=cfg.LOADER.NUM_CLASSES,
        layers=cfg.SPOS.LAYERS,
    )

def _infer_SPOS_CNN():
    from xnas.core.config import cfg
    return infer_SPOS(
        dataset=cfg.LOADER.DATASET,
        resize=cfg.SPOS.RESIZE,
        classes=cfg.LOADER.NUM_CLASSES,
        layers=cfg.SPOS.LAYERS,
        choice=cfg.SPOS.CHOICE,
    )