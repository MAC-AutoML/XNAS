import random
import numpy as np
from scipy import stats

true_list = []
with open('xnas/search_algorithm/RMINAS/sampler/available_archs.txt', 'r') as f:
    true_list = eval(f.readline())

def random_sampling(times):
    sample_list = []
    if times > sum(true_list):
        print('can only sample {} times.'.format(sum(true_list)))
        times = sum(true_list)
    for _ in range(times):
        i = random.randint(0, 15624)
        while (not true_list[i]) or (i in sample_list):
            i = random.randint(0, 15624)
        sample_list.append(i)
    return sample_list

def genostr2array(geno_str):
    # |none~0|+|nor_conv_1x1~0|none~1|+|avg_pool_3x3~0|skip_connect~1|nor_conv_3x3~2|
    OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    _tmp = geno_str.split('|')
    _tmp2 = []
    for i in range(len(_tmp)):
        if i in [1,3,4,6,7,8]:
            _tmp2.append(_tmp[i][:-2])
    _tmp_np = np.array([0]*6)
    for i in range(6):
        _tmp_np[i] = OPS.index(_tmp2[i])
    _tmp_oh = np.zeros((_tmp_np.size, 5))
    _tmp_oh[np.arange(_tmp_np.size),_tmp_np] = 1
    return _tmp_oh

def array2genostr(arr):
    OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    """[[1. 0. 0. 0. 0.]
        [0. 0. 1. 0. 0.]
        [1. 0. 0. 0. 0.]
        [0. 0. 0. 0. 1.]
        [0. 1. 0. 0. 0.]
        [0. 0. 0. 1. 0.]]"""
    idx = [list(i).index(1.) for i in arr]
    op = [OPS[x] for x in idx]
    mixed = '|' + op[0] + '~0|+|' + op[1] + '~0|' + op[2] + '~1|+|' + op[3] + '~0|' + op[4] + '~1|' + op[5] + '~2|'
    return mixed

def base_transform(n, x):
    a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
    b=[]
    while True:
        s=n//x
        y=n%x
        b=b+[y]
        if s==0:
            break
        n=s
    b.reverse()
    zero_arr = [0]*(6-len(b))
    return zero_arr+b

def array_morearch(arr, distance):
    """[[1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0.]
     [0. 0. 0. 1. 0.]]"""
    am = list(arr.argmax(axis=1))  # [0,2,0,4,1,3]
    morearch = []
    if distance == 1:
        for i in range(len(am)):
            for j in range(5):
                if am[i]!=j:
                    _tmp = am[:]
                    _tmp[i] = j
                    _tmp_np = np.array(_tmp)
                    _tmp_oh = np.zeros((_tmp_np.size, 5))
                    _tmp_oh[np.arange(_tmp_np.size),_tmp_np] = 1
                    morearch.append(_tmp_oh)
    else:
        for i in range(15625):
            arr = base_transform(i, 5)
            if distance == 6-sum([arr[i]==am[i] for i in range(6)]):
                _tmp_np = np.array(arr)
                _tmp_oh = np.zeros((_tmp_np.size, 5))
                _tmp_oh[np.arange(_tmp_np.size),_tmp_np] = 1
                morearch.append(_tmp_oh)
    #             morearch.append(arr)
    return morearch



# test_arr = np.array([[1., 0., 0., 0., 0.],
#      [0., 0., 1., 0., 0.],
#      [1., 0., 0., 0., 0.],
#      [0., 0., 0., 0., 1.],
#      [0., 1., 0., 0., 0.],
#      [0., 0., 0., 1., 0.]])
