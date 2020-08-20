import sys
sys.path.append(".")
import argparse
import copy
import os
import pickle
import random
import tqdm
import time
import ConfigSpace
import numpy as np
import matplotlib.pyplot as plt
from nasbench import api

from xnas.search_algorithm.ASNG import ASNG, Dynamic_ASNG
from xnas.search_algorithm.DDPNAS import CategoricalDDPNAS
from xnas.search_algorithm.GridSearch import GridSearch
from xnas.search_algorithm.MDENAS import CategoricalMDENAS
from xnas.search_algorithm.MIGO import MIGO
from xnas.search_algorithm.SNG import SNG, Dynamic_SNG
from xnas.search_space.cell_based_nasben1shot import *
from xnas.search_space.test_function import (EpochSumCategoryTestFunction,
                                             SumCategoryTestFunction)
from xnas.core.utils import index_to_one_hot, one_hot_to_index


def get_optimizer(name, category, step = 4, gamma = 0.9, sample_with_prob=True, utility_function = 'log', utility_function_hyper = 0.4):
    if name == 'DDPNAS':
        return CategoricalDDPNAS(category, 3)
    elif name == 'MDENAS':
        return CategoricalMDENAS(category, 0.01)
    elif name == 'SNG':
        return SNG(categories = category)
    elif name == 'ASNG':
        return ASNG(categories = category)
    elif name == 'dynamic_ASNG':
        return Dynamic_ASNG(categories = category, step = step, pruning = True, sample_with_prob = sample_with_prob)
    elif name == 'dynamic_SNG':
        return Dynamic_SNG(categories = category, step = step,
                           pruning = True, sample_with_prob = sample_with_prob)
    elif name == 'MIGO':
        return MIGO(categories = category, step = step,
                    pruning = True, sample_with_prob = sample_with_prob,
                    utility_function = 'log', utility_function_hyper = utility_function_hyper,
                    momentum = True, gamma = gamma)
    elif name == 'GridSearch':
        return GridSearch(category)
    else:
        raise NotImplementedError


class Reward(object):
    """Computes the fitness of a sampled model by querying NASBench."""
    def __init__(self, space, nasbench, budget):
        self.space = space
        self.nasbench = nasbench
        self.budget = budget

    def compute_reward(self, sample):
        config = ConfigSpace.Configuration(self.space.get_configuration_space(), vector = sample)
        y, c = self.space.objective_function(self.nasbench, config, budget = self.budget)
        fitness = float(y)
        return fitness

    def get_accuracy(self,sample):
        #return test_accuracy of a sample
        config = ConfigSpace.Configuration(self.space.get_configuration_space(), vector = sample)
        adjacency_matrix, node_list = self.space.convert_config_to_nasbench_format(config)
        node_list = [INPUT, *node_list, OUTPUT] if self.space.search_space_number == 3 else [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix = adjacency_list, ops = node_list)
        nasbench_data = self.nasbench.query(model_spec, epochs = self.budget)
        return nasbench_data['test_accuracy']


def run(space = 1, optimizer_name = 'SNG', budget = 108, runing_times = 500, runing_epochs = 200,
        step = 4, gamma = 0.9, save_dir = None, nasbench = None, noise = 0.0, sample_with_prob = True, utility_function = 'log',
        utility_function_hyper = 0.4):
    print('##### Search Space {} #####'.format(space))
    search_space = eval('SearchSpace{}()'.format(space))
    cat_variables = []
    cs = search_space.get_configuration_space()
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
            cat_variables.append(len(h.choices))

    # get category using cat_variables
    category = cat_variables

    # distribution_optimizer = Category_DDPNAS.CategoricalDDPNAS(category, 3)
    distribution_optimizer = get_optimizer(optimizer_name, category, step = step, gamma = gamma,
                                           sample_with_prob = sample_with_prob, utility_function = utility_function,
                                           utility_function_hyper = utility_function_hyper)
    
    # path to save the test_accuracy
    file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'.format(optimizer_name, str(space), str(runing_epochs),
                                                                    str(step), str(gamma), str(noise),
                                                                    str(sample_with_prob), utility_function, str(utility_function_hyper))
    file_name = os.path.join(save_dir, file_name)
    nb_reward = Reward(search_space, nasbench, budget)
    
    record = {
        'validation_accuracy': np.zeros([runing_times, runing_epochs]) - 1,
        'test_accuracy': np.zeros([runing_times, runing_epochs]) - 1,
    }
    last_test_accuracy = np.zeros([runing_times])
    running_time_interval = np.zeros([runing_times, runing_epochs])
    test_accuracy = 0
    for i in tqdm.tqdm(range(runing_times)):


        for j in range(runing_epochs):
            start_time = time.time()
            if hasattr(distribution_optimizer, 'training_finish') or j == (runing_epochs - 1):
                last_test_accuracy[i] = test_accuracy
            if hasattr(distribution_optimizer, 'training_finish'):
                if distribution_optimizer.training_finish:
                    break

            sample = distribution_optimizer.sampling()
            sample_index = one_hot_to_index(np.array(sample))
            validation_accuracy = nb_reward.compute_reward(sample_index)
            distribution_optimizer.record_information(sample, validation_accuracy)
            distribution_optimizer.update()
            current_best = np.argmax(distribution_optimizer.p_model.theta, axis=1)
            test_accuracy = nb_reward.get_accuracy(current_best)
            record['validation_accuracy'][i, j] = validation_accuracy
            record['test_accuracy'][i, j] = test_accuracy
            end_time = time.time()
            running_time_interval[i, j] = end_time - start_time
        
        del distribution_optimizer
        distribution_optimizer = get_optimizer(optimizer_name, category, step=step, gamma=gamma,
                                               sample_with_prob=sample_with_prob, utility_function=utility_function,
                                               utility_function_hyper=utility_function_hyper)

    np.savez(file_name, record['test_accuracy'], running_time_interval)
    return distribution_optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", help = "search space name in [1,2,3]",type = int, default = 1)
    parser.add_argument("--optimizer", help = "dicrete level", type = str, default = 'SNG')
    parser.add_argument("--step", help = "pruning step", type = int, default = 4)
    parser.add_argument("--gamma", help = "gamma value", type = float, default = 0.9)
    parser.add_argument("--noise", help = "noise std", type = float, default = 0.0)
    parser.add_argument("-uh", "--utility_function_hyper",
                        help = "the factor of utility_function", type = float, default = 0.4)
    parser.add_argument("-ut", "--utility_function_type", help = "the type of utility_function", type = str, default = 'log')
    parser.add_argument("-sp", "--sample_with_prob",  action = 'store_true')
    args = parser.parse_args()

    #get nasbench
    nasbench_path = '/PATH/TO/NASBENCH/nasbench_full.tfrecord'
    nasbench = api.NASBench(nasbench_path)

    #get args
    space = args.space
    step = args.step
    gamma = args.gamma
    save_dir = '/PATH/TO/EXPERIMENT'
    optimizer_name = args.optimizer

    print("space = {}, step = {}, gamma = {}, optimizer = {}, noise_std = {}, utility_function_hyper = {}, utility_function_type = {}, sample_with_prob = {}".format(
        str(space), str(step), str(gamma), optimizer_name, str(args.noise), str(args.utility_function_hyper), args.utility_function_type, str(args.sample_with_prob)))
    run(space, optimizer_name = optimizer_name, runing_times = 1, runing_epochs = 100,
        step = step, gamma = gamma, save_dir = save_dir, nasbench = nasbench, noise = args.noise, sample_with_prob = args.sample_with_prob,
        utility_function = args.utility_function_type, utility_function_hyper = args.utility_function_hyper)
