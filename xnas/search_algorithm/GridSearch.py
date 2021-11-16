import numpy as np
from xnas.search_algorithm.utils import Categorical
from xnas.core.utils import index_to_one_hot, one_hot_to_index
import copy

class GridSearch:
    def __init__(self, categories, fresh_size=4, init_theta=None, max_mize=True):

        self.p_model = Categorical(categories)
        self.p_model.C = np.array(self.p_model.C)
        self.valid_d = len(self.p_model.C[self.p_model.C > 1])

        # Refresh theta
        for k in range(self.p_model.d):
            self.p_model.theta[k, 0] = 1
            self.p_model.theta[k, 1:self.p_model.C[k]] = 0
        
        if init_theta is not None:
            self.p_model.theta = init_theta
        
        self.fresh_size = fresh_size
        self.sample = []
        self.objective = []
        self.maxmize = -1 if max_mize else 1
        self.obj_optim = float('inf')
        self.training_finish = False

        # Record point to move
        self.sample_point = self.p_model.theta
        self.point = [self.p_model.d-1, 0]

    def sampling(self):
        return self.sample_point
    
    def record_information(self, sample, objective):
        self.sample.append(sample)
        self.objective.append(objective * self.maxmize)

    def update(self):
        """
        Update sampling by grid search
        e.g.
            categories = [3, 2, 4] 
            sample point as = [1, 1, 1]
                              [0, 0, 0]
                              [0,  , 0]
                              [ ,  , 0]
            point as now searching = [2, 0]
        """
        if len(self.sample) == self.fresh_size:
            # update sample
            if self.point[1] == self.p_model.C[self.point[0]] - 1:
                for i in range(self.point[0] + 1, self.p_model.d):
                    self.sample_point[i] = np.zeros(self.p_model.Cmax)
                    self.sample_point[i][0] = 1
                for j in range(self.point[0], -1, -1):
                    k = np.argmax(self.sample_point[j])
                    if k < self.p_model.C[j] - 1:
                        self.sample_point[j][k] = 0
                        self.sample_point[j][k+1] = 1
                        break
                    else:
                        self.sample_point[j][k] = 0
                        self.sample_point[j][0] = 1
                        if j == 0:
                            self.training_finish = True
                            break
                self.point = [self.p_model.d-1, 0]
            else:
                self.sample_point[self.point[0], self.point[1]] = 0
                self.sample_point[self.point[0], self.point[1]+1] = 1
                self.point[1] += 1

            # update optim and theta
            if min(self.objective) < self.obj_optim:
                self.obj_optim = min(self.objective)
                seq = np.argmin(np.array(self.objective))
                self.p_model.theta = self.sample[seq]

            # update record
            self.sample = []
            self.objective = []