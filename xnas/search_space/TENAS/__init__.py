from typing import List, Text

# __all__ = ['change_key', 'get_cell_based_tiny_net', 'get_search_spaces', 'get_cifar_models', 'get_imagenet_models', \
#                       'obtain_model',
#                       'CellStructure', 'CellArchitectures'
#                       ]

__all__ = ['change_key', 'get_cell_based_tiny_net', 'get_search_spaces', 'CellStructure', 'CellArchitectures']

# useful modules
from xnas.search_space.TENAS.SharedUtils import change_key
from xnas.search_space.TENAS.cell_searchs import CellStructure, CellArchitectures


# Cell-based NAS Models
def get_cell_based_tiny_net(config):
    super_type = getattr(config, 'super_type', 'basic')
    group_names = ['DARTS-V1', 'DARTS-V2']
    if super_type == 'basic' and config.name in group_names:
        from xnas.search_space.TENAS.cell_searchs import nas201_super_nets as nas_super_nets
        try:
            return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space, config.affine, config.track_running_stats, config.depth, config.use_stem)
        except Exception:
            return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space, config.depth, config.use_stem)
    elif super_type == 'nasnet-super':
        from xnas.search_space.TENAS.cell_searchs import nasnet_super_nets as nas_super_nets
        return nas_super_nets[config.name](config.C, config.N, config.steps, config.multiplier, \
                                        config.stem_multiplier, config.num_classes, config.space, config.affine, config.track_running_stats, config.depth, config.use_stem)
    elif config.name == 'infer.tiny':
        from xnas.search_space.TENAS.cell_infers import TinyNetwork
        if hasattr(config, 'genotype'):
            genotype = config.genotype
        elif hasattr(config, 'arch_str'):
            genotype = CellStructure.str2structure(config.arch_str)
        else: raise ValueError('Can not find genotype from this config : {:}'.format(config))
        return TinyNetwork(config.C, config.N, genotype, config.num_classes)
    elif config.name == 'infer.shape.tiny':
        from xnas.search_space.TENAS.shape_infers import DynamicShapeTinyNet
        if isinstance(config.channels, str):
            channels = tuple([int(x) for x in config.channels.split(':')])
        else: channels = config.channels
        genotype = CellStructure.str2structure(config.genotype)
        return DynamicShapeTinyNet(channels, genotype, config.num_classes)
    elif config.name == 'infer.nasnet-cifar':
        from xnas.search_space.TENAS.cell_infers import NASNetonCIFAR
        raise NotImplementedError
    else:
        raise ValueError('invalid network name : {:}'.format(config.name))


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
def get_search_spaces(xtype, name) -> List[Text]:
    if xtype == 'cell':
        from xnas.search_space.TENAS.cell_operations import SearchSpaceNames
        assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
        return SearchSpaceNames[name]
    else:
        raise ValueError('invalid search-space type is {:}'.format(xtype))


# NOTE: these function is never used by author and has lost files.

# def get_cifar_models(config, extra_path=None):
#     super_type = getattr(config, 'super_type', 'basic')
#     if super_type == 'basic':
#         from .CifarResNet      import CifarResNet
#         from .CifarDenseNet    import DenseNet
#         from .CifarWideResNet  import CifarWideResNet
#         if config.arch == 'resnet':
#             return CifarResNet(config.module, config.depth, config.class_num, config.zero_init_residual)
#         elif config.arch == 'densenet':
#             return DenseNet(config.growthRate, config.depth, config.reduction, config.class_num, config.bottleneck)
#         elif config.arch == 'wideresnet':
#             return CifarWideResNet(config.depth, config.wide_factor, config.class_num, config.dropout)
#         else:
#             raise ValueError('invalid module type : {:}'.format(config.arch))
#     elif super_type.startswith('infer'):
#         from .shape_infers import InferWidthCifarResNet
#         from .shape_infers import InferDepthCifarResNet
#         from .shape_infers import InferCifarResNet
#         from .cell_infers import NASNetonCIFAR
#         assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
#         infer_mode = super_type.split('-')[1]
#         if infer_mode == 'width':
#             return InferWidthCifarResNet(config.module, config.depth, config.xchannels, config.class_num, config.zero_init_residual)
#         elif infer_mode == 'depth':
#             return InferDepthCifarResNet(config.module, config.depth, config.xblocks, config.class_num, config.zero_init_residual)
#         elif infer_mode == 'shape':
#             return InferCifarResNet(config.module, config.depth, config.xblocks, config.xchannels, config.class_num, config.zero_init_residual)
#         elif infer_mode == 'nasnet.cifar':
#             genotype = config.genotype
#             if extra_path is not None:  # reload genotype by extra_path
#                 if not osp.isfile(extra_path): raise ValueError('invalid extra_path : {:}'.format(extra_path))
#                 xdata = torch.load(extra_path)
#                 current_epoch = xdata['epoch']
#                 genotype = xdata['genotypes'][current_epoch-1]
#             C = config.C if hasattr(config, 'C') else config.ichannel
#             N = config.N if hasattr(config, 'N') else config.layers
#             return NASNetonCIFAR(C, N, config.stem_multi, config.class_num, genotype, config.auxiliary)
#         else:
#             raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
#     else:
#         raise ValueError('invalid super-type : {:}'.format(super_type))


# def get_imagenet_models(config):
#     super_type = getattr(config, 'super_type', 'basic')
#     if super_type == 'basic':
#         from .ImageNet_ResNet import ResNet
#         from .ImageNet_MobileNetV2 import MobileNetV2
#         if config.arch == 'resnet':
#             return ResNet(config.block_name, config.layers, config.deep_stem, config.class_num, config.zero_init_residual, config.groups, config.width_per_group)
#         elif config.arch == 'mobilenet_v2':
#             return MobileNetV2(config.class_num, config.width_multi, config.input_channel, config.last_channel, 'InvertedResidual', config.dropout)
#         else:
#             raise ValueError('invalid arch : {:}'.format( config.arch ))
#     elif super_type.startswith('infer'): # NAS searched architecture
#         assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
#         infer_mode = super_type.split('-')[1]
#         if infer_mode == 'shape':
#             from .shape_infers import InferImagenetResNet
#             from .shape_infers import InferMobileNetV2
#             if config.arch == 'resnet':
#                 return InferImagenetResNet(config.block_name, config.layers, config.xblocks, config.xchannels, config.deep_stem, config.class_num, config.zero_init_residual)
#             elif config.arch == "MobileNetV2":
#                 return InferMobileNetV2(config.class_num, config.xchannels, config.xblocks, config.dropout)
#             else:
#                 raise ValueError('invalid arch-mode : {:}'.format(config.arch))
#         else:
#             raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
#     else:
#         raise ValueError('invalid super-type : {:}'.format(super_type))


# # Try to obtain the network by config.
# def obtain_model(config, extra_path=None):
#     if config.dataset == 'cifar':
#         return get_cifar_models(config, extra_path)
#     elif config.dataset == 'imagenet':
#         return get_imagenet_models(config)
#     else:
#         raise ValueError('invalid dataset in the model config : {:}'.format(config))
