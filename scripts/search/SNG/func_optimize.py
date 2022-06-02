"""
This file is derived from the original XNAS source code, 
and has not been rigorously tested subsequently.

If you find it useful, please open an issue to tell us.

recoded file path: XNAS/tools/sng_function_optimization.py
"""


import argparse
import os
import numpy as np

from xnas.logger.timer import Timer
from xnas.algorithms.SNG.ASNG import ASNG, Dynamic_ASNG
from xnas.algorithms.SNG.DDPNAS import CategoricalDDPNAS
from xnas.algorithms.SNG.MDENAS import CategoricalMDENAS
from xnas.algorithms.SNG.MIGO import MIGO
from xnas.algorithms.SNG.GridSearch import GridSearch
from xnas.algorithms.SNG.SNG import SNG, Dynamic_SNG



def get_optimizer(name, category, step=4, gamma=0.9, sample_with_prob=True, utility_function='log', utility_function_hyper=0.4):
    if name == 'SNG':
        return SNG(categories=category)
    elif name == 'ASNG':
        return ASNG(categories=category)
    elif name == 'dynamic_SNG':
        return Dynamic_SNG(categories=category, step=step,
                           pruning=True, sample_with_prob=sample_with_prob)
    elif name == 'dynamic_ASNG':
        return Dynamic_ASNG(categories=category, step=step, pruning=True, sample_with_prob=sample_with_prob)
    elif name == 'DDPNAS':
        return CategoricalDDPNAS(category, 100, gamma=0.8, theta_lr=0.01)
    elif name == 'MDENAS':
        return CategoricalMDENAS(category, 0.01)
    elif name == 'MIGO':
        return MIGO(categories=category, step=step, lam=6,
                    pruning=True, sample_with_prob=sample_with_prob,
                    utility_function='log', utility_function_hyper=utility_function_hyper,
                    momentum=True, gamma=gamma, dynamic_sampling=False)
    elif name == 'GridSearch':
        return GridSearch(category)
    else:
        raise NotImplementedError


def run(M=10, N=10, func='rastrigin', optimizer_name='SNG', running_times=500, running_epochs=200,
        step=4, gamma=0.9, save_dir=None, noise=0.0, sample_with_prob=True, utility_function='log',
        utility_function_hyper=0.4):
    category = [M]*N
    epoc_fun = 'linear'
    test_fun = EpochSumCategory(
        category, epoch_func=epoc_fun, func=func, noise_std=noise)

    distribution_optimizer = get_optimizer(optimizer_name, category, step=step, gamma=gamma,
                                           sample_with_prob=sample_with_prob,
                                           utility_function=utility_function,
                                           utility_function_hyper=utility_function_hyper)
    file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'.format(optimizer_name, str(N), str(M), str(running_epochs),
                                                                 epoc_fun, func, str(step), str(
                                                                     gamma), str(noise),
                                                                 str(sample_with_prob), utility_function, str(utility_function_hyper))

    file_name = os.path.join(save_dir, file_name)

    record = {
        'objective': np.zeros([running_times, running_epochs]) - 1,
        'l2_distance': np.zeros([running_times, running_epochs]) - 1
    }
    last_l2_distance = np.zeros([running_times])
    running_time_interval = np.zeros([running_times, running_epochs])
    _distance = 100
    run_timer = Timer()
    for i in range(running_times):
        for j in range(running_epochs):
            run_timer.tic()
            if hasattr(distribution_optimizer, 'training_finish') or j == (running_epochs - 1):
                last_l2_distance[i] = _distance
            if hasattr(distribution_optimizer, 'training_finish'):
                if distribution_optimizer.training_finish:
                    break
            sample = distribution_optimizer.sampling()
            objective = test_fun.objective_function(sample)
            distribution_optimizer.record_information(sample, objective)
            distribution_optimizer.update()

            current_best = np.argmax(
                distribution_optimizer.p_model.theta, axis=1)
            _distance = test_fun.l2_distance(current_best)
            record['l2_distance'][i, j] = objective
            record['objective'][i, j] = _distance
            run_timer.toc()
            running_time_interval[i, j] = run_timer.diff
        test_fun.re_new()
        del distribution_optimizer
        # print(_distance)
        distribution_optimizer = get_optimizer(optimizer_name, category, step=step, gamma=gamma,
                                               sample_with_prob=sample_with_prob,
                                               utility_function=utility_function,
                                               utility_function_hyper=utility_function_hyper)
    np.savez(file_name, record['l2_distance'], running_time_interval)
    return distribution_optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="function dimension", type=int, default=10)
    parser.add_argument("--M", help="dicrete level", type=int, default=10)
    parser.add_argument(
        "--func", help="test functions in [rastrigin, index_sum, rosenbrock]", type=str, default='rastrigin')
    parser.add_argument("--optimizer", help="dicrete level",
                        type=str, default='DDPNAS')
    parser.add_argument("--step", help="pruning step", type=int, default=4)
    parser.add_argument("--gamma", help="gamma value", type=float, default=0.9)
    parser.add_argument("--noise", help="noise std", type=float, default=0.0)
    parser.add_argument("-uh", "--utility_function_hyper",
                        help="the factor of utility_function", type=float, default=0.4)
    parser.add_argument("-ut", "--utility_function_type",
                        help="the type of utility_function", type=str, default='log')
    parser.add_argument("-sp", "--sample_with_prob",  action='store_true')
    parser.add_argument("--save_dir", help="save directory", type=str,
                        default='experiment/sng_function_optimization')

    args = parser.parse_args()
    func = args.func
    step = args.step
    gamma = args.gamma
    save_dir = args.save_dir
    optimizer_name = args.optimizer

    print("N={}, M={}, function={}, step={}, gamma={}, optimizer={}, noise_std={}, utility_function_hyper={}, utility_function_type={}, sample_with_prob={}".format(
        str(args.N), str(args.M), func, str(step), str(gamma), optimizer_name, str(args.noise), str(args.utility_function_hyper), args.utility_function_type, str(args.sample_with_prob)))
    run(M=args.M, N=args.N, func=func, optimizer_name=optimizer_name, running_times=500, running_epochs=1000,
        step=step, gamma=gamma, save_dir=save_dir, noise=args.noise, sample_with_prob=args.sample_with_prob,
        utility_function=args.utility_function_type, utility_function_hyper=args.utility_function_hyper)



def rastrigin_function(input):
    f_x = 10. * input.shape[0]
    for i in input:
        f_x += i**2 - 10 * np.cos(2*np.pi*i)
    return f_x


def rosenbrock_function(input):
    f_x = 0
    for i in range(input.shape[0]-1):
        f_x += 100 * (input[i+1] - input[i]**2)**2 + (1 - input[i]) ** 2
    return f_x


class EpochSumCategory():
    def __init__(self, category, epoch_func='quad', func='index_sum', noise_std=0.0):
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
        self.noise_std = noise_std

    def re_new(self):
        self.epoch_recoder = []
        for i in range(len(self.category)):
            self.epoch_recoder.append(np.zeros(self.category[i]))
        self.epoch_recoder = np.array(self.epoch_recoder)

    def input_trans(self, input):
        return (input - self.bias) * self.scale

    def objective_function(self, sample):
        _sample = []
        for i in sample:
            _sample.append(np.argmax(i))
        sample = np.array(_sample)
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
        random_noise = np.random.randn() * self.noise_std + 1.
        return result * self.maxmize * random_noise

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
