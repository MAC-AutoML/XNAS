import numpy as np
from search_algorithm.Category_dist import Categorical


class CategoricalMDENAS:
    def __init__(self, category, learning_rate):
        self.p_model = Categorical(categories=category)
        self.leaning_rate = learning_rate
        self.information_recoder = {'epoch': np.zeros((self.p_model.d, self.p_model.Cmax)),
                                    'performance': np.zeros((self.p_model.d, self.p_model.Cmax))}

    def sampling(self):
        return self.p_model.sampling()

    def sampling_index(self):
        return self.p_model.sampling_index()

    def record_information(self, sample, performance):
        for i in range(len(sample)):
            self.information_recoder['epoch'][i, sample[i]] += 1
            self.information_recoder['performance'][i, sample[i]] = performance

    def update(self):

        # update the probability
        for edges_index in range(self.p_model.d):
            for i in range(self.p_model.Cmax):
                for j in range(i+1, self.p_model.Cmax):
                    if (self.information_recoder['epoch'][edges_index, i]
                        >= self.information_recoder['epoch'][edges_index, j])\
                            and (self.information_recoder['performance'][edges_index, i]
                                 < self.information_recoder['performance'][edges_index, j]):
                        if self.p_model.theta[edges_index, i] > self.leaning_rate:
                            self.p_model.theta[edges_index, i] -= self.leaning_rate
                            self.p_model.theta[edges_index, j] += self.leaning_rate
                        else:
                            self.p_model.theta[edges_index, j] += self.p_model.theta[edges_index, i]
                            self.p_model.theta[edges_index, i] = 0

                    if (self.information_recoder['epoch'][edges_index, i]
                        <= self.information_recoder['epoch'][edges_index, j]) \
                        and (self.information_recoder['performance'][edges_index, i]
                             > self.information_recoder['performance'][edges_index, j]):
                        if self.p_model.theta[edges_index, j] > self.leaning_rate:
                            self.p_model.theta[edges_index, j] -= self.leaning_rate
                            self.p_model.theta[edges_index, i] += self.leaning_rate
                        else:
                            self.p_model.theta[edges_index, i] += self.p_model.theta[edges_index, j]
                            self.p_model.theta[edges_index, j] = 0


