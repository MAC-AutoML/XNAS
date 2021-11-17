import ConfigSpace
import numpy as np

from xnas.search_space.cellbased_1shot1_ops import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3, OUTPUT_NODE

from nasbench import api


def Eval_nasbench1shot1(theta, search_space, logger):
    nasbench1shot1_path = 'benchmark/nasbench_full.tfrecord'
    nasbench = api.NASBench(nasbench1shot1_path)
    
    current_best = np.argmax(theta, axis=1)
    config = ConfigSpace.Configuration(
        search_space.search_space.get_configuration_space(), vector=current_best)
    adjacency_matrix, node_list = search_space.search_space.convert_config_to_nasbench_format(
        config)
    node_list = [INPUT, *node_list, OUTPUT] if search_space.search_space.search_space_number == 3 else [
        INPUT, *node_list, CONV1X1, OUTPUT]
    adjacency_list = adjacency_matrix.astype(np.int).tolist()
    model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
    nasbench_data = nasbench.query(model_spec, epochs=108)
    
    logger.info("test accuracy = {}".format(nasbench_data['test_accuracy']))
