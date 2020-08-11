import copy
import itertools
import random
from abc import abstractmethod

import ConfigSpace
import numpy as np
from nasbench import api
from collections import namedtuple

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT_NODE = 6

PRIMITIVES = [
    'maxpool3x3',
    'conv3x3-bn-relu',
    'conv1x1-bn-relu'
]


def upscale_to_nasbench_format(adjacency_matrix):
    """
    The search space uses only 4 intermediate nodes, rather than 5 as used in nasbench
    This method adds a dummy node to the graph which is never used to be compatible with nasbench.
    :param adjacency_matrix:
    :return:
    """
    return np.insert(
        np.insert(adjacency_matrix,
                  5, [0, 0, 0, 0, 0, 0], axis=1),
        5, [0, 0, 0, 0, 0, 0, 0], axis=0)


def parent_combinations_old(adjacency_matrix, node, n_parents=2):
    """Get all possible parent combinations for the current node."""
    if node != 1:
        # Parents can only be nodes which have an index that is lower than the current index,
        # because of the upper triangular adjacency matrix and because the index is also a
        # topological ordering in our case.
        return itertools.combinations(np.argwhere(adjacency_matrix[:node, node] == 0).flatten(),
                                      n_parents)  # (e.g. (0, 1), (0, 2), (1, 2), ...
    else:
        return [[0]]


def parent_combinations(node, num_parents):
    if node == 1 and num_parents == 1:
        return [(0,)]
    else:
        return list(itertools.combinations(list(range(int(node))), num_parents))


class SearchSpace:
    def __init__(self, search_space_number, num_intermediate_nodes):
        self.search_space_number = search_space_number
        self.num_intermediate_nodes = num_intermediate_nodes
        self.num_parents_per_node = {}

        self.run_history = []

    @abstractmethod
    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        """Based on given connectivity pattern create the corresponding adjacency matrix."""
        pass

    def sample(self, with_loose_ends, upscale=True):
        if with_loose_ends:
            adjacency_matrix_sample = self._sample_adjacency_matrix_with_loose_ends()
        else:
            adjacency_matrix_sample = self._sample_adjacency_matrix_without_loose_ends(
                adjacency_matrix=np.zeros([self.num_intermediate_nodes + 2, self.num_intermediate_nodes + 2]),
                node=self.num_intermediate_nodes + 1)
            assert self._check_validity_of_adjacency_matrix(adjacency_matrix_sample), 'Incorrect graph'

        if upscale and self.search_space_number in [1, 2]:
            adjacency_matrix_sample = upscale_to_nasbench_format(adjacency_matrix_sample)
        return adjacency_matrix_sample, random.choices(PRIMITIVES, k=self.num_intermediate_nodes)

    def _sample_adjacency_matrix_with_loose_ends(self):
        parents_per_node = [random.sample(list(itertools.combinations(list(range(int(node))), num_parents)), 1) for
                            node, num_parents in self.num_parents_per_node.items()][2:]
        parents = {
            '0': [],
            '1': [0]
        }
        for node, node_parent in enumerate(parents_per_node, 2):
            parents[str(node)] = node_parent
        adjacency_matrix = self._create_adjacency_matrix_with_loose_ends(parents)
        return adjacency_matrix

    def _sample_adjacency_matrix_without_loose_ends(self, adjacency_matrix, node):
        req_num_parents = self.num_parents_per_node[str(node)]
        current_num_parents = np.sum(adjacency_matrix[:, node], dtype=np.int)
        num_parents_left = req_num_parents - current_num_parents
        sampled_parents = \
            random.sample(list(parent_combinations_old(adjacency_matrix, node, n_parents=num_parents_left)), 1)[0]
        for parent in sampled_parents:
            adjacency_matrix[parent, node] = 1
            adjacency_matrix = self._sample_adjacency_matrix_without_loose_ends(adjacency_matrix, parent)
        return adjacency_matrix

    @abstractmethod
    def generate_adjacency_matrix_without_loose_ends(self, **kwargs):
        """Returns every adjacency matrix in the search space without loose ends."""
        pass

    def convert_config_to_nasbench_format(self, config):
        parents = {node: config["choice_block_{}_parents".format(node)] for node in
                   list(self.num_parents_per_node.keys())[1:]}
        parents['0'] = []
        adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(parents)
        ops = [config["choice_block_{}_op".format(node)] for node in list(self.num_parents_per_node.keys())[1:-1]]
        return adjacency_matrix, ops

    def get_configuration_space(self):
        cs = ConfigSpace.ConfigurationSpace()

        for node in list(self.num_parents_per_node.keys())[1:-1]:
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("choice_block_{}_op".format(node),
                                                                        [CONV1X1, CONV3X3, MAXPOOL3X3]))

        for choice_block_index, num_parents in list(self.num_parents_per_node.items())[1:]:
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter(
                    "choice_block_{}_parents".format(choice_block_index),
                    parent_combinations(node=choice_block_index, num_parents=num_parents)))
        return cs

    def generate_search_space_without_loose_ends(self):
        # Create all possible connectivity patterns
        for iter, adjacency_matrix in enumerate(self.generate_adjacency_matrix_without_loose_ends()):
            print(iter)
            # Print graph
            # Evaluate every possible combination of node ops.
            n_repeats = int(np.sum(np.sum(adjacency_matrix, axis=1)[1:-1] > 0))
            for combination in itertools.product([CONV1X1, CONV3X3, MAXPOOL3X3], repeat=n_repeats):
                # Create node labels
                # Add some op as node 6 which isn't used, here conv1x1
                ops = [INPUT]
                combination = list(combination)
                for i in range(5):
                    if np.sum(adjacency_matrix, axis=1)[i + 1] > 0:
                        ops.append(combination.pop())
                    else:
                        ops.append(CONV1X1)
                assert len(combination) == 0, 'Something is wrong'
                ops.append(OUTPUT)

                # Create nested list from numpy matrix
                nasbench_adjacency_matrix = adjacency_matrix.astype(np.int).tolist()

                # Assemble the model spec
                model_spec = api.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=nasbench_adjacency_matrix,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=ops)

                yield adjacency_matrix, ops, model_spec

    def _generate_adjacency_matrix(self, adjacency_matrix, node):
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            yield adjacency_matrix
        else:
            req_num_parents = self.num_parents_per_node[str(node)]
            current_num_parents = np.sum(adjacency_matrix[:, node], dtype=np.int)
            num_parents_left = req_num_parents - current_num_parents

            for parents in parent_combinations_old(adjacency_matrix, node, n_parents=num_parents_left):
                # Make copy of adjacency matrix so that when it returns to this stack
                # it can continue with the unmodified adjacency matrix
                adjacency_matrix_copy = copy.copy(adjacency_matrix)
                for parent in parents:
                    adjacency_matrix_copy[parent, node] = 1
                    for graph in self._generate_adjacency_matrix(adjacency_matrix=adjacency_matrix_copy, node=parent):
                        yield graph

    def _create_adjacency_matrix(self, parents, adjacency_matrix, node):
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            return adjacency_matrix
        else:
            for parent in parents[str(node)]:
                adjacency_matrix[parent, node] = 1
                if parent != 0:
                    adjacency_matrix = self._create_adjacency_matrix(parents=parents, adjacency_matrix=adjacency_matrix,
                                                                     node=parent)
            return adjacency_matrix

    def _create_adjacency_matrix_with_loose_ends(self, parents):
        # Create the adjacency_matrix on a per node basis
        adjacency_matrix = np.zeros([len(parents), len(parents)])
        for node, node_parents in parents.items():
            for parent in node_parents:
                adjacency_matrix[parent, int(node)] = 1
        return adjacency_matrix

    def _check_validity_of_adjacency_matrix(self, adjacency_matrix):
        """
        Checks whether a graph is a valid graph in the search space.
        1. Checks that the graph is non empty
        2. Checks that every node has the correct number of inputs
        3. Checks that if a node has outgoing edges then it should also have incoming edges
        4. Checks that input node is connected
        5. Checks that the graph has no more than 9 edges
        :param adjacency_matrix:
        :return:
        """
        # Check that the graph contains nodes
        num_intermediate_nodes = sum(np.array(np.sum(adjacency_matrix, axis=1) > 0, dtype=int)[1:-1])
        if num_intermediate_nodes == 0:
            return False

        # Check that every node has exactly the right number of inputs
        col_sums = np.sum(adjacency_matrix[:, :], axis=0)
        for col_idx, col_sum in enumerate(col_sums):
            if col_sum > 0:
                if col_sum != self.num_parents_per_node[str(col_idx)]:
                    return False

        # Check that if a node has outputs then it should also have incoming edges (apart from zero)
        col_sums = np.sum(np.sum(adjacency_matrix, axis=0) > 0)
        row_sums = np.sum(np.sum(adjacency_matrix, axis=1) > 0)
        if col_sums != row_sums:
            return False

        # Check that the input node is always connected. Otherwise the graph is disconnected.
        row_sum = np.sum(adjacency_matrix, axis=1)
        if row_sum[0] == 0:
            return False

        # Check that the graph returned has no more than 9 edges.
        num_edges = np.sum(adjacency_matrix.flatten())
        if num_edges > 9:
            return False

        return True


class SearchSpace1(SearchSpace):
    def __init__(self):
        super(SearchSpace1, self).__init__(search_space_number=1, num_intermediate_nodes=4)
        """
        SEARCH SPACE 1
        """
        self.num_parents_per_node = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 2,
            '4': 2,
            '5': 2
        }
        if sum(self.num_parents_per_node.values()) > 9:
            raise ValueError('Each nasbench cell has at most 9 edges.')

        self.test_min_error = 0.05448716878890991
        self.valid_min_error = 0.049278855323791504

    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        adjacency_matrix = self._create_adjacency_matrix(parents, adjacency_matrix=np.zeros([6, 6]),
                                                         node=OUTPUT_NODE - 1)
        # Create nasbench compatible adjacency matrix
        return upscale_to_nasbench_format(adjacency_matrix)

    def create_nasbench_adjacency_matrix_with_loose_ends(self, parents):
        return upscale_to_nasbench_format(self._create_adjacency_matrix_with_loose_ends(parents))

    def generate_adjacency_matrix_without_loose_ends(self):
        for adjacency_matrix in self._generate_adjacency_matrix(adjacency_matrix=np.zeros([6, 6]),
                                                                node=OUTPUT_NODE - 1):
            yield upscale_to_nasbench_format(adjacency_matrix)

    def objective_function(self, nasbench, config, budget=108):
        adjacency_matrix, node_list = super(SearchSpace1, self).convert_config_to_nasbench_format(config)
        # adjacency_matrix = upscale_to_nasbench_format(adjacency_matrix)
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = nasbench.query(model_spec, epochs=budget)
        return nasbench_data['validation_accuracy'], nasbench_data['training_time']

    def generate_with_loose_ends(self):
        for _, parent_node_3, parent_node_4, output_parents in itertools.product(
                *[itertools.combinations(list(range(int(node))), num_parents) for node, num_parents in
                  self.num_parents_per_node.items()][2:]):
            parents = {
                '0': [],
                '1': [0],
                '2': [0, 1],
                '3': parent_node_3,
                '4': parent_node_4,
                '5': output_parents
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(parents)
            yield adjacency_matrix


Architecture = namedtuple('Architecture', ['adjacency_matrix', 'node_list'])


class Model(object):
    """A class representing a model.

    It holds two attributes: `arch` (the simulated architecture) and `accuracy`
    (the simulated accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:  (as in the original code)
      arch: the architecture as an int representing a bit-string of length `DIM`.
          As a result, the integers are required to be less than `2**DIM`. They
          can be visualized as strings of 0s and 1s by calling `print(model)`,
          where `model` is an instance of this class.
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self):
        self.arch = None
        self.validation_accuracy = None
        self.test_accuracy = None
        self.training_time = None
        self.budget = None

    def update_data(self, arch, nasbench_data, budget):
        self.arch = arch
        self.validation_accuracy = nasbench_data['validation_accuracy']
        self.test_accuracy = nasbench_data['test_accuracy']
        self.training_time = nasbench_data['training_time']
        self.budget = budget

    def query_nasbench(self, nasbench, sample, search_space=None):
        config = ConfigSpace.Configuration(
            search_space.get_configuration_space(), vector=sample
        )
        adjacency_matrix, node_list = search_space.convert_config_to_nasbench_format(config)
        if type(search_space) == SearchSpace3:
            node_list = [INPUT, *node_list, OUTPUT]
        else:
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)

        nasbench_data = nasbench.query(model_spec)
        self.arch = Architecture(adjacency_matrix=adjacency_matrix,
                                 node_list=node_list)
        self.validation_accuracy = nasbench_data['validation_accuracy']
        self.test_accuracy = nasbench_data['test_accuracy']
        self.training_time = nasbench_data['training_time']


class SearchSpace2(SearchSpace):
    def __init__(self):
        self.search_space_number = 2
        self.num_intermediate_nodes = 4
        super(SearchSpace2, self).__init__(search_space_number=self.search_space_number,
                                           num_intermediate_nodes=self.num_intermediate_nodes)
        """
        SEARCH SPACE 2
        """
        self.num_parents_per_node = {
            '0': 0,
            '1': 1,
            '2': 1,
            '3': 2,
            '4': 2,
            '5': 3
        }
        if sum(self.num_parents_per_node.values()) > 9:
            raise ValueError('Each nasbench cell has at most 9 edges.')

        self.test_min_error = 0.057592153549194336
        self.valid_min_error = 0.051582515239715576

    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        adjacency_matrix = self._create_adjacency_matrix(parents, adjacency_matrix=np.zeros([6, 6]),
                                                         node=OUTPUT_NODE - 1)
        # Create nasbench compatible adjacency matrix
        return upscale_to_nasbench_format(adjacency_matrix)

    def create_nasbench_adjacency_matrix_with_loose_ends(self, parents):
        return upscale_to_nasbench_format(self._create_adjacency_matrix_with_loose_ends(parents))

    def generate_adjacency_matrix_without_loose_ends(self):
        for adjacency_matrix in self._generate_adjacency_matrix(adjacency_matrix=np.zeros([6, 6]),
                                                                node=OUTPUT_NODE - 1):
            yield upscale_to_nasbench_format(adjacency_matrix)

    def objective_function(self, nasbench, config, budget=108):
        adjacency_matrix, node_list = super(SearchSpace2, self).convert_config_to_nasbench_format(config)
        # adjacency_matrix = upscale_to_nasbench_format(adjacency_matrix)
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = nasbench.query(model_spec, epochs=budget)

        # record the data to history
        architecture = Model()
        arch = Architecture(adjacency_matrix=adjacency_matrix,
                            node_list=node_list)
        architecture.update_data(arch, nasbench_data, budget)
        self.run_history.append(architecture)

        return nasbench_data['validation_accuracy'], nasbench_data['training_time']

    def generate_with_loose_ends(self):
        for parent_node_2, parent_node_3, parent_node_4, output_parents in itertools.product(
                *[itertools.combinations(list(range(int(node))), num_parents) for node, num_parents in
                  self.num_parents_per_node.items()][2:]):
            parents = {
                '0': [],
                '1': [0],
                '2': parent_node_2,
                '3': parent_node_3,
                '4': parent_node_4,
                '5': output_parents
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(parents)
            yield adjacency_matrix


class SearchSpace3(SearchSpace):
    def __init__(self):
        self.search_space_number = 3
        self.num_intermediate_nodes = 5
        super(SearchSpace3, self).__init__(search_space_number=self.search_space_number,
                                           num_intermediate_nodes=self.num_intermediate_nodes)
        """
        SEARCH SPACE 3
        """
        self.num_parents_per_node = {
            '0': 0,
            '1': 1,
            '2': 1,
            '3': 1,
            '4': 2,
            '5': 2,
            '6': 2
        }
        if sum(self.num_parents_per_node.values()) > 9:
            raise ValueError('Each nasbench cell has at most 9 edges.')

        self.test_min_error = 0.05338543653488159
        self.valid_min_error = 0.04847759008407593

    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        # Create nasbench compatible adjacency matrix
        adjacency_matrix = self._create_adjacency_matrix(parents, adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE)
        return adjacency_matrix

    def create_nasbench_adjacency_matrix_with_loose_ends(self, parents):
        return self._create_adjacency_matrix_with_loose_ends(parents)

    def generate_adjacency_matrix_without_loose_ends(self):
        for adjacency_matrix in self._generate_adjacency_matrix(adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE):
            yield adjacency_matrix

    def generate_with_loose_ends(self):
        for parent_node_2, parent_node_3, parent_node_4, parent_node_5, output_parents in itertools.product(
                *[itertools.combinations(list(range(int(node))), num_parents) for node, num_parents in
                  self.num_parents_per_node.items()][2:]):
            parents = {
                '0': [],
                '1': [0],
                '2': parent_node_2,
                '3': parent_node_3,
                '4': parent_node_4,
                '5': parent_node_5,
                '6': output_parents
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(parents)
            yield adjacency_matrix

    def objective_function(self, nasbench, config, budget=108):
        adjacency_matrix, node_list = super(SearchSpace3, self).convert_config_to_nasbench_format(config)
        node_list = [INPUT, *node_list, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = nasbench.query(model_spec, epochs=budget)
        return nasbench_data['validation_accuracy'], nasbench_data['training_time']
