from xnas.search_space.cell_based import DartsCNN
from xnas.search_space.cell_based import NASBench201CNN

import torch
import numpy as np
import torch.nn.functional as F

if __name__ == "__main__":
    # dartscnn test
    search_net = DartsCNN()
    _random_architecture_weight = torch.randn([search_net.num_edges * 2, len(search_net.basic_op_list)])
    _input = torch.randn([2, 3, 32, 32])
    _out_put = search_net(_input, search_net)
    pass
