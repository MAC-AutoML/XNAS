import numpy as np
from xnas.search_algorithm.utils import Categorical
from xnas.core.utils import index_to_one_hot, one_hot_to_index
import copy


class RAND:
    """
    Random Sample for Categorical Distribution
    """

    def __init__(self, categories, delta_init=1., opt_type="best", init_theta=None, max_mize=True):
        # Categorical distribution
        self.p_model = Categorical(categories)
        # valid dimension size
        self.p_model.C = np.array(self.p_model.C)

        if init_theta is not None:
            self.p_model.theta = init_theta

        self.sample_list = []
        self.obj_list = []
        self.max_mize = -1 if max_mize else 1

        self.select = opt_type
        self.best_object = 1e10 * self.max_mize

    def record_information(self, sample, objective):
        self.sample_list.append(sample)
        self.obj_list.append(objective*self.max_mize)

    def sampling(self):
        """
        Draw a sample from the categorical distribution (one-hot)
        Sample one archi at once
        """
        c = np.zeros(self.p_model.theta.shape, dtype=np.bool)
        for i, upper in enumerate(self.p_model.C):
            j = np.random.randint(upper)
            c[i, j] = True
        return c

    def sampling_index(self):
        return one_hot_to_index(np.array(self.sampling()))

    def mle(self):
        """
        Get most likely categorical variables (one-hot)
        """
        m = self.p_model.theta.argmax(axis=1)
        x = np.zeros((self.p_model.d, self.p_model.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x

    def update(self):
        objective = np.array(self.obj_list[-1])
        sample = np.array(self.sample_list[-1])
        if (self.select == 'best'):
            if (objective > self.best_object):
                self.best_object = objective
                # refresh theta to best one
                self.p_model.theta = np.array(sample)
        else:
            raise NotImplementedError
        self.sample_list = []
        self.obj_list = []
