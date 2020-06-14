import numpy as np
import pdb
# test function


def rastrigin_function(input):
    f_x = 10. * input.shape[0]
    for i in input:
        f_x += i**2 - 10 * np.cos(2*np.pi*i)
    return f_x


def rosenbrock_function(input):
    f_x = 0
    for i in range(input.shape[0]-1):
        f_x += 100 * (input[i+1] - input[i]**2)**2 + (1 - input[i] ** 2)
    return f_x


class TestFunction:
    def optimal_value(self):
        raise NotImplementedError

    def objective_function(self, x):
        raise NotImplementedError

    def l2_distance(self, sample):
        raise NotImplementedError

    def re_new(self):
        pass


class SumCategoryTestFunction(TestFunction):
    def __init__(self, category):
        self.category = category

    def objective_function(self, sample):
        return np.sum(sample)

    def epoch_objective_function(self, sample, epoch):
        return np.sum(sample) * np.log10(epoch)

    def optimal_value(self):
        return np.array(self.category) - 1

    def l2_distance(self, sample):
        return np.sum((self.optimal_value() - sample) ** 2)


class EpochSumCategoryTestFunction(TestFunction):
    def __init__(self, category, epoch_func='quad', func='index_sum'):
        assert epoch_func in ['quad', 'linear', 'exp', 'constant']
        assert func in ['index_sum', 'rastrigin', 'rosenbrock']
        self.category = category
        self.epoch_recoder = []
        for i in range(len(self.category)):
            self.epoch_recoder.append(np.zeros(self.category[i]))
        self.epoch_recoder = np.array(self.epoch_recoder)
        self.epoch_func = epoch_func
        self.func = func
        if self.func == 'index_sum':
            self.scale = 1.
            self.bias = 1.
        elif self.func == 'rastrigin':
            # search range is [-5.12, 5.12]
            self.scale = 10./float(self.category[0])
            self.bias = int(self.category[0]/2)
        elif self.func == 'rosenbrock':
            # search range is (-inf, +inf)
            self.scale = 1.
            self.bias = int(self.category[0]/2)
        self.maxmize = 1 if self.func == 'index_sum' else -1

    def re_new(self):
        self.epoch_recoder = []
        for i in range(len(self.category)):
            self.epoch_recoder.append(np.zeros(self.category[i]))
        self.epoch_recoder = np.array(self.epoch_recoder)

    def input_trans(self, input):
        return (input - self.bias) * self.scale

    def objective_function(self, sample):

        sample = self.input_trans(sample)
        epoch = []
        for i in range(sample.shape[0]):
            self.epoch_recoder[i, int(sample[i])] += 1
            epoch.append(self.epoch_recoder[i, int(sample[i])])
        epoch = np.array(epoch)
        if self.epoch_func == 'exp':
            epoch = np.exp(epoch)
        elif self.epoch_func == 'linear':
            epoch = epoch
        elif self.epoch_func == 'quad':
            epoch = epoch ** 5
        elif self.epoch_func == 'constant':
            epoch = np.ones(epoch.shape)
        else:
            raise NotImplementedError
        # ['index_sum', 'rastrigin', 'rosenbrock ']
        if self.func == 'index_sum':
            result = np.sum(sample)
        elif self.func == 'rastrigin':
            result = rastrigin_function(sample)
            # result *= np.random.choice(epoch)
        else:
            result = rosenbrock_function(sample)

        max_epoch = np.max(epoch)
        min_epoch = np.min(epoch)
        random_seed = np.random.randn() * (max_epoch - min_epoch) + 1.
        result *= random_seed
        return result * self.maxmize

    def optimal_value(self):
        if self.func == 'index_sum':
            opt_value = np.array(self.category) - 1
        elif self.func == 'rastrigin':
            opt_value = np.zeros(len(self.category))
        else:
            opt_value = np.ones(len(self.category))

        return opt_value

    def l2_distance(self, sample):
        sample = self.input_trans(sample)
        return np.sum((self.optimal_value() - sample) ** 2) / len(self.category)

