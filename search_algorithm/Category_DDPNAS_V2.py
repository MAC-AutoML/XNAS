import numpy as np
from search_algorithm.Category_dist import Categorical
import copy
from scipy.special import softmax
import random


class CategoricalDDPNASV2:
    def __init__(self, category, steps):
        self.p_model = Categorical(categories=category)
        # how many steps to pruning the distribution
        self.steps = steps
        self.current_step = 1
        self.ignore_index = []
        self.sample_index = []
        self.pruned_index = []
        self.val_performance = []
        self.sample = []
        self.init_record()
        self.score_decay = 0.5
        self.training_finish = False
        self.training_epoch = self.get_training_epoch()
        self.non_param_index = [0, 1, 2, 3]
        self.param_index = [4, 5, 6, 7]
        self.non_param_index_num = len(self.non_param_index)
        self.pruned_index_num = len(self.param_index)
        self.non_param_index_count = [0] * self.p_model.d
        self.param_index_count = [0] * self.p_model.d

    def init_record(self):
        for i in range(self.p_model.d):
            self.ignore_index.append([])
            self.sample_index.append(list(range(self.p_model.Cmax)))
            self.pruned_index.append([])
        self.val_performance.append(np.zeros([self.p_model.d, self.p_model.Cmax]))

    def get_training_epoch(self):
        return self.steps * sum(list(range(self.p_model.Cmax)))

    def sampling(self):
        return self.sampling_index()

    def sampling_index(self):
        sample = []
        for i in range(self.p_model.d):
            sample.append(random.choice(self.sample_index[i]))
            if len(self.sample_index[i]) > 0:
                self.sample_index[i].remove(sample[i])
        return np.array(sample)

    def sample_with_constrains(self):
        pass

    def record_information(self, sample, performance):
        self.sample = sample
        for i in range(self.p_model.d):
            self.val_performance[-1][i, self.sample[i]] = performance

    def update_sample_index(self):
        for i in range(self.p_model.d):
            self.sample_index[i] = list(set(range(self.p_model.Cmax)) - set(self.pruned_index[i]))

    def update(self):
        # when a search epoch for operations is over and not to the total search epoch
        if len(self.sample_index[0]) == 0:
            if self.current_step < self.steps:
                self.current_step += 1
                # append new val performance
                self.val_performance.append(np.zeros([self.p_model.d, self.p_model.Cmax]))
            # when the total search is down
            else:
                self.current_step += 1
                expectation = np.zeros([self.p_model.d, self.p_model.Cmax])
                for i in range(self.steps):
                    # multi 100 to ignore 0
                    expectation += softmax(self.val_performance[i] * 100, axis=1)
                expectation = expectation / float(self.steps)
                self.p_model.theta = expectation + self.score_decay * self.p_model.theta
                # prune the index
                pruned_weight = copy.deepcopy(self.p_model.theta)
                for index in range(self.p_model.d):
                    if not len(self.pruned_index[index]) == 0:
                        pruned_weight[index, self.pruned_index[index]] = np.nan
                    pruned_index = np.nanargmin(pruned_weight[index, :])
                    if self.non_param_index_count[index] == 3 and pruned_index in self.non_param_index:
                        pruned_weight[index, pruned_index] = np.nan
                        pruned_index = np.nanargmin(pruned_weight[index, :])
                    if self.param_index_count[index] == 3 and pruned_index in self.param_index:
                        pruned_weight[index, pruned_index] = np.nan
                        pruned_index = np.nanargmin(pruned_weight[index, :])
                    if pruned_index in self.param_index:
                        self.param_index_count[index] += 1
                    if pruned_index in self.non_param_index:
                        self.non_param_index_count[index] += 1
                    self.pruned_index[index].append(pruned_index)
                    self.p_model.theta[index, pruned_index] = 0
                self.p_model.theta /= np.sum(self.p_model.theta, axis=1)[:, np.newaxis]
                if self.param_index_count[0] == 3 and self.non_param_index_count[0] == 3:
                    self.training_finish = True
                self.current_step = 1
                # init val_performance
                self.val_performance = []
                self.val_performance.append(np.zeros([self.p_model.d, self.p_model.Cmax]))
            self.update_sample_index()


