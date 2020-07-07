from xnas.search_space.cell_based import DartsCNN
from xnas.search_space.cell_based import NASBench201CNN
from xnas.core.timer import Timer

import torch
import numpy as np
import torch.nn.functional as F

if __name__ == "__main__":
    # dartscnn test
    time_ = Timer()
    print("Testing darts CNN")
    search_net = DartsCNN()
    _random_architecture_weight = torch.randn(
        [search_net.num_edges * 2, len(search_net.basic_op_list)])
    _input = torch.randn([2, 3, 32, 32])
    time_.tic()
    _out_put = search_net(_input, _random_architecture_weight)
    time_.toc()
    print(_out_put.shape)
    print(time_.average_time)
    time_.reset()
    _random_one_hot = torch.Tensor(np.eye(len(search_net.basic_op_list))[
                                   np.random.choice(len(search_net.basic_op_list), search_net.num_edges * 2)])
    _input = torch.randn([2, 3, 32, 32])
    time_.tic()
    _out_put = search_net(_input, _random_one_hot)
    time_.toc()
    print(_out_put.shape)
    print(time_.average_time)
    pass
