import numpy as np
import torch
from collections import namedtuple

basic_op_list = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none']
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def random_suggest():
    sample = np.zeros((14, 7)) # 14边，7op
    node_ids = np.asarray([np.random.choice(range(x,x+i+2), size=2, replace=False) for i, x in enumerate((0,2,5,9))]).ravel() # 选择哪8个边
    op = np.random.multinomial(1,[1/7.]*7, size=8) # 8条选择的边、7个有意义op
    sample[node_ids] = op
    return sample

def ransug2alpha(suggest_sample):
    b = np.c_[suggest_sample, np.zeros(14)]
    return torch.from_numpy(np.r_[b,b])

def geno2147array(genotype):
    """
    Genotype(normal=[[('max_pool_3x3', 0), ('dil_conv_3x3', 1)], [('max_pool_3x3', 0), ('dil_conv_5x5', 1)], [('avg_pool_3x3', 1), ('dil_conv_3x3', 0)], [('dil_conv_3x3', 0), ('sep_conv_3x3', 3)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('dil_conv_3x3', 1)], [('max_pool_3x3', 0), ('dil_conv_5x5', 1)], [('avg_pool_3x3', 1), ('dil_conv_3x3', 0)], [('dil_conv_3x3', 0), ('sep_conv_3x3', 3)]], reduce_concat=range(2, 6))
    """
    genotype = eval(genotype)
    sample = np.zeros([28, 7])
    norm_gene = genotype[0]
    reduce_gene = genotype[2]
    num_select = list(range(2, 6))
    for j, _gene in enumerate([norm_gene, reduce_gene]):
        for i, node in enumerate(_gene):
            for op in node:
                op_name = op[0]
                op_id = op[1]
                if i == 0:
                    true_id = op_id + j * 14
                else:
                    if i == 1:
                        _temp = num_select[0]
                    else:
                        _temp = sum(num_select[0:i])
                    true_id = op_id + _temp + j * 14
                sample[true_id, basic_op_list.index(op_name)] = 1
#     for i in range(28):
#         if np.sum(sample[i, :]) == 0:
#             sample[i, 7] = 1
    return sample[0:14]
    