import argparse
import os
from functools import partial
import torch
import time


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


def _parser(parser):
    # dataset parser
    parser.add_argument('--dataset', required=False, default='CIFAR10', help='CIFAR10 / MNIST / FashionMNIST / ImageNet')
    parser.add_argument('--data_path', required=False, default='/userhome/data/cifar10',
                        help='data path')
    parser.add_argument('--image_size', type=int, default=0, help='The size of the input Image')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for the data set')
    parser.add_argument('--datset_split', type=int, default=10, help='dataset split')
    parser.add_argument('--workers', type=int, default=4, help='# of workers')
    # network parser:darts
    parser.add_argument('--init_channels', type=int, default=16)
    parser.add_argument('--layers', type=int, default=8, help='# of layers')
    # node is fixed in most case
    parser.add_argument('--n_nodes', type=int, default=4, help='# nodes in each cell')
    # network parser:proxy_less_nas
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width_mult', type=float, default=1.0)

    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--bn_eps', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0)
    # training parser
    parser.add_argument('--w_lr', type=float, default=0.1, help='lr for weights')
    parser.add_argument('--w_lr_min', type=float, default=0.0001, help='minimum learning rate')
    parser.add_argument('--w_lr_step', type=int, default=50, help='lr for weights')
    parser.add_argument('--w_lr_gamma', type=float, default=0.1, help='minimum lr for weights')
    parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
    parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                        help='weight decay for weights')
    parser.add_argument('--w_grad_clip', type=float, default=5.,
                        help='gradient clipping for weights')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                                                    '`all` indicates use all gpus.')
    parser.add_argument('--epochs', type=int, default=1000, help='# of training epochs')
    # for one shot NAS
    parser.add_argument('--warm_up_epochs', type=int, default=0, help='# of training epochs')
    # random seed
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='cudnn switch')
    # optimizer parser
    parser.add_argument('--pruning_step', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.8)


class SearchConfig(BaseConfig):
    @staticmethod
    def build_parser():
        parser = get_parser("Search config")
        parser.add_argument('--name', default='dynamic_SNG_V3', required=False,
                            help='MDENAS / DDPNAS / SNG/ ASNG/ dynamic_ASNG/ dynamic_SNG_V3/others will be comming soon')
        parser.add_argument('--search_space', default='proxyless', required=False,
                            help='darts/ proxyless/ google/ofa others will be comming soon')
        parser.add_argument('--sub_name', default='', required=False)
        
        _parser(parser)
        return parser

    def __init__(self):
        parser = SearchConfig.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
        time_str = time.asctime(time.localtime()).replace(' ', '_')
        #'w_lr': '0.2', 'w_momentum': '0.8', 'w_weight_decay': '0.003', 'w_lr_step': '30', 'datset_split': '20',
        # 'warm_up_epochs': '14', 'pruning_step': '5', 'gamma': '0.7'
        name_componment = [
                           'seed_' + str(self.seed),
                           'w_lr' + str(self.w_lr),
                           'w_momentum_' + str(self.w_momentum),
                           'w_weight_decay' + str(self.w_weight_decay),
                           'w_lr_step_' + str(self.w_lr_step),
                           'data_split_' + str(self.datset_split),
                           'warm_up_epochs_' + str(self.warm_up_epochs),
                           ]

        # name_componment = [
        #                    'width_multi_' + str(self.width_mult),
        #                    'epochs_' + str(self.epochs),
        #                    'data_split_' + str(self.datset_split),
        #                    'warm_up_epochs_' + str(self.warm_up_epochs),
        #                    'lr_' + str(self.w_lr),
        #                    ]
        if 'dynamic' in self.name or 'DDPNAS' in self.name:
            name_componment += ['pruning_step_' + str(self.pruning_step),
                                'gamma_' + str(self.gamma)]
        if self.search_space == 'darts':
            name_componment += ['init_channels' + str(self.init_channels),
                                'layers' + str(self.layers),
                                'n_nodes' + str(self.n_nodes)]
        else:
            name_componment += ['width_multi_' + str(self.width_mult)]
        name_str = ''
        for i in name_componment:
            name_str += i + '_'
        name_str += time_str
        self.path = os.path.join('experiments', self.name,
                                 self.search_space,
                                 self.dataset, name_str)
        # self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)
        self.image_size = None if self.image_size == 0 else self.image_size
        if self.search_space in ['proxyless', 'google', 'ofa']:
            self.conv_candidates = [
                '3x3_MBConv3', '3x3_MBConv6',
                '5x5_MBConv3', '5x5_MBConv6',
                '7x7_MBConv3', '7x7_MBConv6',
            ]
        self.network_info_path = os.path.join(self.path, 'network_info')
