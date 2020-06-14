import numpy as np
import itertools
from utils import utils
from utils import genotypes
import os
import copy
import json
import glob
import pdb


def un_pack_list(input_list, n_node=4):
    assert len(input_list) == sum([i + 2 for i in range(n_node)])
    out_list = []
    _pre = 0
    for i in range(n_node):
        out_list.append(input_list[_pre:_pre + i + 2])
        _pre = _pre + i + 2
    return out_list


def get_best_gene(height_constraint_graph, _skip_connection_selection_list, max_indexes, _prob):
    # get the best
    best_expectation = 0
    best_gene = None
    for graph in height_constraint_graph:
        for _skip_connection_selection in _skip_connection_selection_list:
            expectation = 0
            graph_operation_index = []
            for node_index in range(len(graph)):
                node_operation_index = []
                # _node: record the input node
                _node = graph[node_index]
                left_node_index, right_node_index = (2 * node_index), 2 * node_index + 1

                # get the node operation index
                if left_node_index in _skip_connection_selection:
                    # print('noparam%d'%max_indexes[node_index][_node[0]][0])
                    node_operation_index.append(max_indexes[node_index][_node[0]][0])
                else:
                    # print('param%d'%max_indexes[node_index][_node[0]][1])
                    node_operation_index.append(max_indexes[node_index][_node[0]][1])

                if right_node_index in _skip_connection_selection:
                    # print('noparam%d'%max_indexes[node_index][_node[0]][0])
                    node_operation_index.append(max_indexes[node_index][_node[1]][0])
                else:
                    # print('param%d'%max_indexes[node_index][_node[0]][1])
                    node_operation_index.append(max_indexes[node_index][_node[1]][1])

                expectation += _prob[node_index][_node[0]][node_operation_index[0]]
                expectation += _prob[node_index][_node[1]][node_operation_index[1]]
                graph_operation_index.append(node_operation_index)

            if expectation > best_expectation:
                best_expectation = expectation
                # generate the gene

                # print('graph')
                # print(graph)
                # print('graph_operation_index')
                # print(graph_operation_index)
                best_gene = genotypes.parse_graph_and_operation(graph, graph_operation_index)
    return best_gene, best_expectation


def get_network(probability, reduce_constrain=True, height_constraint=2, skip_connection_constraint=2):
    max_indexs = []
    for i in probability:
        _prob = copy.deepcopy(i)
        noparm_prob = _prob[0:3]
        parm_prob = _prob[3:7]
        _ = []
        # print('prob:')
        # print(_prob)
        _.append(np.argmax(noparm_prob))
        # print('noparam %d'%_[0])
        _.append(np.argmax(parm_prob) + 3)
        # print('param %d'%_[1])
        max_indexs.append(_)
    # print('---')
    pre_product = []
    this_product = []
    for i in range(4):
        this_combination = itertools.combinations(list(range(i + 2)), 2)
        this_combination = list(this_combination)
        if i == 0:
            pre_product = this_combination
            this_product = this_combination
        else:
            this_product = list(itertools.product(pre_product, this_combination))
            pre_product = this_product
    # flat tuple
    this_product = [(a, b, c, d) for ((a, b), c), d in this_product]
    # height constraint
    height_constraint_graph = []
    for graph in this_product:
        height_list = []
        for node in graph:
            left_height = 0 if node[0] in [0, 1] else height_list[node[0] - 2]
            right_height = 0 if node[1] in [0, 1] else height_list[node[1] - 2]
            node_height = left_height if left_height >= right_height else right_height
            node_height += 1
            height_list.append(node_height)
        graph_height = max(height_list)
        if graph_height <= height_constraint:
            height_constraint_graph.append(graph)
    # unpack the prob
    n_node = 4
    n_edges = sum([i + 2 for i in list(range(4))])
    prob_norm = utils.darts_weight_unpack(probability[0:n_edges], n_node)
    prob_reduce = utils.darts_weight_unpack(probability[n_edges:], n_node)
    max_indexes_norm = un_pack_list(max_indexs[0: n_edges])
    max_indexes_reduce = un_pack_list(max_indexs[n_edges:])
    _skip_connection_selection_list = list(itertools.combinations(list(range(n_node * 2)), skip_connection_constraint))
    # norm cell
    best_norm_gene, best_norm_expectation = get_best_gene(height_constraint_graph, _skip_connection_selection_list,
                                                          max_indexes_norm, prob_norm)
    if reduce_constrain:
        best_reduce_gene, best_reduction_expectation = get_best_gene(height_constraint_graph,
                                                                     _skip_connection_selection_list,
                                                                     max_indexes_reduce, prob_reduce)
        best_expectation = (best_norm_expectation + best_reduction_expectation) / 2.
    else:
        best_reduce_gene = genotypes.parse_numpy(prob_reduce, k=2)
        best_expectation = best_norm_expectation
    concat = range(2, 2 + 4)  # concat all intermediate nodes
    best_geno = genotypes.Genotype(normal=best_norm_gene, normal_concat=concat,
                                   reduce=best_reduce_gene, reduce_concat=concat)
    return best_geno, best_expectation


def get_gene_by_prob(path, prob):
    height_constraints = [1, 2, 3, 4]
    skip_connection_constraints = [2, 4, 6, 8]
    reductions = [True, False]
    for _height_constraint in height_constraints:
        for _skip_connection_constraint in skip_connection_constraints:
            for _reduction in reductions:
                save_text_name = str(_height_constraint) + '_' + str(_skip_connection_constraint) + \
                                 '_' + str(_reduction) + '.txt'
                print(save_text_name)
                file = open(os.path.join(path, save_text_name), 'w+')
                gen, _ = get_network(prob, reduce_constrain=_reduction,
                                     height_constraint=_height_constraint,
                                     skip_connection_constraint=_skip_connection_constraint)
                file.write(str(gen) + "\n")
                file.close()


def get_gene_with_skip_connection_constraints(prob, skip_constraint=2, reduction=False):
    height_constraints = [1, 2, 3, 4]
    best_expectation = 0.
    best_gene = None
    for _height_constraint in height_constraints:
        gen, current_expectation = get_network(prob, reduce_constrain=reduction,
                                               height_constraint=_height_constraint,
                                               skip_connection_constraint=skip_constraint)
        if current_expectation > best_expectation:
            best_expectation = current_expectation
            best_gene = gen
    return best_gene


def get_gene_by_dir(dir_name):
    # dir_name = '/userhome/project/DDPNAS_V2/experiment'
    dirs = os.listdir(dir_name)
    height_constraints = [1, 2, 3, 4]
    skip_connection_constraints = [2, 4, 6, 8]
    reductions = [True, False]
    # get the prob list
    prob_list = []
    for _dir in dirs:
        _path = os.path.join(dir_name, _dir)
        prob_path = os.path.join(_path, 'probability.npy')
        if os.path.exists(prob_path):
            prob = np.load(prob_path)
            prob_list.append(prob)
    for _height_constraint in height_constraints:
        for _skip_connection_constraint in skip_connection_constraints:
            for _reduction in reductions:
                save_text_name = str(_height_constraint) + '_' + str(_skip_connection_constraint) + \
                                 '_' + str(_reduction) + '.txt'
                print(save_text_name)
                file = open(os.path.join(dir_name, 'txt', save_text_name), 'w+')
                for _prob in prob_list:
                    gen = get_network(_prob, reduce_constrain=_reduction,
                                      height_constraint=_height_constraint,
                                      skip_connection_constraint=_skip_connection_constraint)
                    file.write(str(gen) + "\n")
                file.close()


def getw(w):
    return int(w * 100)


def get_path(weight, constraint, FLOPS):
    # n=20
    # weight=np.array([np.random.rand(8) for i in range(n)])
    # constraint=np.array([np.random.rand(8) for i in range(n)])
    # print(weight.shape)
    # print(constraint.shape)
    # FLOPS=10
    n = weight.shape[0]
    c = weight.shape[1]
    max_weight = 0
    for i in range(n):
        max_weight += int(np.max(weight[i]) * 100)  # 保留小数点后两位
    # print('max_weight:', max_weight)      #weigth最大可能取的值，作为状态的上限

    dp = [[FLOPS * 10 for i in range(max_weight + 5)] for i in range(n)]
    # 定义dp[n][max_weight] dp[i][j]表示第i个节点，在权值和为j的情况下，能取到的最小的限制值
    pre = np.zeros((n, max_weight + 5), int)  # 记录pre[i][j]的前驱节点，是由上个节点哪个状态(j)转移到的
    chose = np.zeros((n, max_weight + 5), int)  # 记录dp[i][j]这个状态在节点i是选择了哪条路径

    for i in range(c):  # 对第0个节点初始化
        w = getw(weight[0][i])
        dp[0][w] = constraint[0][i]
        pre[0][w] = -1  # -1表示没有前驱
        chose[0][w] = i
    ans = 0  # 记录限制下可以取得的最大权值
    endk = 0  # 记录最后一个节点取了哪条path
    for i in range(1, n):  # 遍历每个分组
        for j in range(max_weight + 1):  # 遍历每个容量，即权值
            for k in range(c):  # 遍历当前分组种可选的物品,即路径
                w = getw(weight[i][k])
                if (j >= w):
                    if (dp[i][j] > dp[i - 1][j - w] + constraint[i][k]):
                        dp[i][j] = dp[i - 1][j - w] + constraint[i][k]
                        pre[i][j] = j - w  # dp[i][j]由dp[i-1][j-w]转移过来，所以前驱是(i-1,j-w)
                        chose[i][j] = k  # 选择了第k个物品
                if i == n - 1:
                    if dp[i][j] <= FLOPS and j > ans:
                        ans = j
                        endk = k
    path = []
    path.append(endk)
    nowj = ans
    nownode = n - 1
    while (pre[nownode][nowj] != -1):  # 根据记录的前驱不断回溯找到每个分组选择了哪个物品，即每个节点选择了哪条边
        nowj = pre[nownode][nowj]
        nownode -= 1
        # print(dp[nownode][nowj])
        # print(chose[nownode][nowj])
        path.append(chose[nownode][nowj])

    path = path[::-1]
    # print('max_weight:',ans)
    # print('path:',path)
    return path


def get_MB_network(dir_name, flops_constraint=600, name=None):
    flops_constraint = flops_constraint
    if not os.path.exists(os.path.join(dir_name, 'probability.npy')):
        return None
    flops_list = json.load(open(os.path.join(dir_name, 'flops.json')))
    super_net = json.load(open(os.path.join(dir_name, 'supernet.json')))
    prob = np.load(os.path.join(dir_name, 'probability.npy'))
    total_flops = 0
    total_flops += (flops_list['first_conv_flpos'] + flops_list['feature_mix_layer_flops'] +
                    flops_list['classifier_flops'] + flops_list['block_flops'][0][0])
    if 'final_expand_layer_flops' in flops_list.keys():
        total_flops += flops_list['final_expand_layer_flops']
    total_flops = total_flops
    block_flops = np.array(flops_list['block_flops'][1:])
    assert block_flops.shape[0] == prob.shape[0]
    # print(prob)
    path = get_path(prob, block_flops, flops_constraint - total_flops)
    _net = copy.deepcopy(super_net)
    assert len(path) == len(_net['blocks']) - 1
    for i in range(len(path)):
        _net['blocks'][i + 1]['mobile_inverted_conv'] = \
            _net['blocks'][i + 1]['mobile_inverted_conv']['selection'][path[i]]
    if name is None:
        save_path = os.path.join(dir_name, str(flops_constraint) + '.json')
    else:
        save_path = os.path.join(dir_name, name + '.json')
    json.dump(_net, open(save_path, 'a+'))
    return path


def get_constraint():
    network_info_path_list = glob.glob(
        '/userhome/project/Auto_NAS_V2/experiments/dynamic_SNG_V3/darts/*/seed*/network_info')
    # print(network_info_path_list)
    network_info_path_list = sorted(network_info_path_list)
    for network_info_path in network_info_path_list:
        theta = np.load(os.path.join(network_info_path, 'probability.npy'))
        # print(theta)
        print(network_info_path)
        for i in [2, 3, 4]:
            print(i)
            a = get_gene_with_skip_connection_constraints(theta, skip_constraint=i, reduction=False)
            print(a)


if __name__ == '__main__':
    pass
    # get_constraint()
    # network_info_path = '/userhome/project/Auto_NAS_V2/experiments/DDPNAS_V2/darts/cifar10/width_multi_0.0_epochs_1000_data_split_10_warm_up_epochs_0_lr_0.1_init_channels16_layers8_n_nodes4_Tue_Feb_25_09:17:17_2020/network_info'
    # network_info_path = '/userhome/code/'
    # network_info_path_list = glob.glob('/userhome/project/Auto_NAS_V2/experiments/dynamic_SNG_V3/darts/*/seed*/network_info')
    # # print(network_info_path_list)
    # network_info_path_list = sorted(network_info_path_list)
    # for network_info_path in network_info_path_list:
    #     theta = np.load(os.path.join(network_info_path, 'probability.npy'))
    #     # print(theta)
    #     print(network_info_path)
    #     for i in [2, 3, 4]:
    #         print(i)
    #         a = get_gene_with_skip_connection_constraints(theta, skip_constraint=i, reduction=False)
    #         print(a)
    # get_gene_by_prob(network_info_path, theta)
    # get_network(theta, reduce_constrain=False)
    # for i in [400, 500, 600]:
    #     a = get_MB_network('/userhome/project/Auto_NAS_V2/experiment/dynamic_SNG_V3/'
    #                        'ofa__epochs_200_data_split_10_warm_up_epochs_0_pruning_step_3_Wed_Jan_22_11:52:11_2020/'
    #                        'network_info', i)
    #     print(a)

