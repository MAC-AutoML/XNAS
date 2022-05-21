from collections import namedtuple
from .ops import *


basic_op_list = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none']


def geno_from_alpha(theta):
    Genotype = namedtuple(
        'Genotype', 'normal normal_concat reduce reduce_concat')
    theta_norm = darts_weight_unpack(
        theta[0:14], 4)
    theta_reduce = darts_weight_unpack(
        theta[14:], 4)
    gene_normal = parse_from_numpy(
        theta_norm, k=2, basic_op_list=basic_op_list)
    gene_reduce = parse_from_numpy(
        theta_reduce, k=2, basic_op_list=basic_op_list)
    concat = range(2, 6)  # concat all intermediate nodes
    return Genotype(normal=gene_normal, normal_concat=concat,
                    reduce=gene_reduce, reduce_concat=concat)

def reformat_DARTS(genotype):
    """
    format genotype for DARTS-like
    from:
        Genotype(normal=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 0)], [('sep_conv_3x3', 2), ('max_pool_3x3', 1)], [('sep_conv_3x3', 3), ('dil_conv_3x3', 2)], [('dil_conv_5x5', 4), ('dil_conv_5x5', 3)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1)], [('max_pool_3x3', 0), ('dil_conv_5x5', 2)], [('max_pool_3x3', 0), ('sep_conv_5x5', 1)], [('dil_conv_5x5', 4), ('max_pool_3x3', 0)]], reduce_concat=range(2, 6))
    to:
        Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
    """
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    _normal = []
    _reduce = []
    for i in genotype.normal:
        for j in i:
            _normal.append(j)
    for i in genotype.reduce:
        for j in i:
            _reduce.append(j)
    _normal_concat = [i for i in genotype.normal_concat]
    _reduce_concat = [i for i in genotype.reduce_concat]
    r_genotype = Genotype(
        normal=_normal,
        normal_concat=_normal_concat,
        reduce=_reduce,
        reduce_concat=_reduce_concat
    )
    return r_genotype
