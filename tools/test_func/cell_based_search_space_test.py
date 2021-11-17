import numpy as np
import torch
import torch.nn.functional as F

from xnas.core.timer import Timer
from xnas.search_space.cellbased_DARTS_cnn import DartsCNN
from xnas.search_space.cellbased_NASBench201_cnn import NASBench201CNN


def basic_darts_cnn_test():
    # dartscnn test
    time_ = Timer()
    print("Testing darts CNN")
    search_net = DartsCNN().cuda()
    _random_architecture_weight = torch.randn(
        [search_net.num_edges * 2, len(search_net.basic_op_list)]).cuda()
    _input = torch.randn([2, 3, 32, 32]).cuda()
    time_.tic()
    _out_put = search_net(_input, _random_architecture_weight)
    time_.toc()
    print(_out_put.shape)
    print(time_.average_time)
    time_.reset()
    _random_one_hot = torch.Tensor(np.eye(len(search_net.basic_op_list))[
                                   np.random.choice(len(search_net.basic_op_list), search_net.num_edges * 2)]).cuda()
    _input = torch.randn([2, 3, 32, 32]).cuda()
    time_.tic()
    _out_put = search_net(_input, _random_one_hot)
    time_.toc()
    print(_out_put.shape)
    print(time_.average_time)


def basic_nas_bench_201_cnn_test():
    #  nas_bench_201 test
    time_ = Timer()
    print("Testing nas bench 201 CNN")
    search_net = NASBench201CNN()
    _random_architecture_weight = torch.randn(
        [search_net.num_edges, len(search_net.basic_op_list)])
    _input = torch.randn([2, 3, 32, 32])
    time_.tic()
    _out_put = search_net(_input, _random_architecture_weight)
    time_.toc()
    print(_out_put.shape)
    print(time_.average_time)
    time_.reset()
    _random_one_hot = torch.Tensor(np.eye(len(search_net.basic_op_list))[
                                   np.random.choice(len(search_net.basic_op_list), search_net.num_edges)])
    _input = torch.randn([2, 3, 32, 32])
    time_.tic()
    _out_put = search_net(_input, _random_one_hot)
    time_.toc()
    print(_out_put.shape)
    print(time_.average_time)


if __name__ == "__main__":
    basic_darts_cnn_test()
    basic_nas_bench_201_cnn_test()
    pass
