import argparse
import copy
import os
import pickle
import random

import ConfigSpace
import numpy as np

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


def get_optimizer(name, category, step=4, gamma=0.9, sample_with_prob=True, utility_function='log', utility_function_hyper=0.4):
    if name == 'DDPNAS':
        return CategoricalDDPNAS(category, 3)
    elif name == 'MDENAS':
        return CategoricalMDENAS(category, 0.01)
    elif name == 'SNG':
        return SNG(categories=category)
    elif name == 'ASNG':
        return ASNG(categories=category)
    elif name == 'dynamic_ASNG':
        return Dynamic_ASNG(categories=category, step=step, pruning=True, sample_with_prob=sample_with_prob)
    elif name == 'dynamic_SNG':
        return Dynamic_SNG(categories=category, step=step,
                           pruning=True, sample_with_prob=sample_with_prob)
    elif name == 'MIGO':
        return MIGO(categories=category, step=step,
                    pruning=True, sample_with_prob=sample_with_prob,
                    utility_function='log', utility_function_hyper=utility_function_hyper,
                    momentum=True, gamma=gamma)
    elif name == 'GridSearch':
        return GridSearch(category)
    else:
        raise NotImplementedError


class Reward(object):
    """Computes the fitness of a sampled model by querying NASBench."""

    def __init__(self, space):
        self.space = space

    def compute_reward(self, sample):
        config = ConfigSpace.Configuration(self.space.get_configuration_space(), vector=sample)
        y, c = self.space.objective_function(nasbench, config)
        fitness = float(y)
        return fitness


def run(space=1, optimizer_name='SNG', runing_times=500, runing_epochs=200,
        step=4, gamma=0.9, save_dir=None, noise=0.0, sample_with_prob=True, utility_function='log',
        utility_function_hyper=0.4):
    print('##### Search Space {} #####'.format(space))
    search_space = eval('SearchSpace{}()'.format(space))
    nb_reward = Reward(search_space)
    cat_variables = []
    cs = search_space.get_configuration_space()
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
            cat_variables.append(len(h.choices))

    # distribution_optimizer = Category_DDPNAS.CategoricalDDPNAS(category, 3)
    distribution_optimizer = get_optimizer(optimizer_name, category, step=step, gamma=gamma,
                                           sample_with_prob=sample_with_prob, utility_function=utility_function,
                                           utility_function_hyper=utility_function_hyper)
