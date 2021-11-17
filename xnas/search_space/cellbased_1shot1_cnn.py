from xnas.search_space.cellbased_1shot1_ops import *
from xnas.core.utils import index_to_one_hot, one_hot_to_index


class NASBench1shot1Cell(nn.Module):

    def __init__(self, steps, C_prev, C, layer, search_space):
        super(NASBench1shot1Cell, self).__init__()
        self._steps = steps

        self._choice_blocks = nn.ModuleList()
        self._bns = nn.ModuleList()
        self.search_space = search_space

        self._input_projections = nn.ModuleList()
        C_in = C_prev if layer == 0 else C_prev * steps

        for i in range(self._steps):
            choice_block = ChoiceBlock(C_in=C)
            self._choice_blocks.append(choice_block)
            self._input_projections.append(ConvBnRelu(C_in=C_in, C_out=C, kernel_size=1, stride=1, padding=0))

        self._input_projections.append(ConvBnRelu(C_in=C_in, C_out=C * self._steps, kernel_size=1, stride=1, padding=0))

    def forward(self, s0, weights, output_weights, input_weights):

        states = []

        # Loop through the choice blocks of each cell
        for choice_block_idx in range(self._steps):
            if input_weights is not None:
                if (choice_block_idx == 0) or (choice_block_idx == 1 and type(self.search_space) == SearchSpace1):
                    input_weight = None
                else:
                    input_weight = input_weights.pop(0)

            s = self._choice_blocks[choice_block_idx](inputs=[self._input_projections[choice_block_idx](s0), *states],
                                                      input_weights=input_weight, weights=weights[choice_block_idx])
            states.append(s)
        input_to_output_edge = self._input_projections[-1](s0)
        # assert (len(input_weights) == 0, 'Something went wrong here.')

        if output_weights is None:
            tensor_list = states
        else:
            tensor_list = [w * t for w, t in zip(output_weights[0][1:], states)]

        return output_weights[0][0] * input_to_output_edge + torch.cat(tensor_list, dim=1)


class NASBench1shot1CNN(nn.Module):

    def __init__(self, C, num_classes, layers, search_space, steps=4):
        super(NASBench1shot1CNN, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self.search_space = search_space

        # In NASBench the stem has 128 output channels
        C_curr = C
        self.stem = ConvBnRelu(C_in=3, C_out=C_curr, kernel_size=3, stride=1)

        self.cells = nn.ModuleList()
        C_prev = C_curr
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # Double the number of channels after each down-sampling step
                # Down-sample in forward method
                C_curr *= 2
            cell = NASBench1shot1Cell(steps=self._steps, C_prev=C_prev, C=C_curr, layer=i, search_space=search_space)
            self.cells += [cell]
            C_prev = C_curr
        self.postprocess = StdConv(C_in=C_prev * self._steps, C_out=C_curr, kernel_size=1, stride=1, padding=0,
                                      affine=False)

        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input, sample):
        # get weights by sample
        sample_index = one_hot_to_index(np.array(sample))
        config = ConfigSpace.Configuration(self.search_space.get_configuration_space(), vector=sample_index)
        adjacency_matrix, node_list = self.search_space.convert_config_to_nasbench_format(config)
        arch_parameters = get_weights_from_arch((adjacency_matrix, node_list),
                                                self._steps, self.search_space.search_space_number)
        # NASBench only has one input to each cell
        s0 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                # Perform down-sampling by factor 1/2
                s0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(s0)

            # get mixed_op_weights
            mixed_op_weights = arch_parameters[0]

            # get output_weights
            output_weights = arch_parameters[1]

            # get input_weights
            input_weights = [alpha for alpha in arch_parameters[2:]]

            s0 = cell(s0, mixed_op_weights, output_weights, input_weights)

        # Include one more preprocessing step here
        s0 = self.postprocess(s0)  # [N, C_max * (steps + 1), w, h] -> [N, C_max, w, h]

        # Global Average Pooling by averaging over last two remaining spatial dimensions
        out = s0.view(*s0.shape[:2], -1).mean(-1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def genotype(self, theta):
        sample = np.argmax(theta, axis=1)
        config = ConfigSpace.Configuration(self.search_space.get_configuration_space(), vector=sample)
        adjacency_matrix, node_list = self.search_space.convert_config_to_nasbench_format(config)
        if self.search_space.search_space_number == 3:
            node_list = [INPUT, *node_list, OUTPUT]
        else:
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        result = "adjacency_matrix:"+str(adjacency_matrix)+"node_list:"+str(node_list)
        return result


# bulid API


def _NASbench1shot1_1():
    from xnas.core.config import cfg
    return NASBench1shot1CNN(C=cfg.SPACE.CHANNEL,
                   num_classes=cfg.SEARCH.NUM_CLASSES,
                   layers=9,
                   search_space=SearchSpace1(),
                   steps=4)


def _NASbench1shot1_2():
    from xnas.core.config import cfg
    return NASBench1shot1CNN(C=cfg.SPACE.CHANNEL,
                   num_classes=cfg.SEARCH.NUM_CLASSES,
                   layers=9,
                   search_space=SearchSpace2(),
                   steps=4)


def _NASbench1shot1_3():
    from xnas.core.config import cfg
    return NASBench1shot1CNN(C=cfg.SPACE.CHANNEL,
                   num_classes=cfg.SEARCH.NUM_CLASSES,
                   layers=9,
                   search_space=SearchSpace3(),
                   steps=5)
