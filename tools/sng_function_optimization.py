import sys
import argparse
import os
import time
from typing_extensions import runtime
import numpy as np
import tqdm

from xnas.core.timer import Timer
from xnas.search_algorithm.ASNG import ASNG, Dynamic_ASNG
from xnas.search_algorithm.DDPNAS import CategoricalDDPNAS
from xnas.search_algorithm.MDENAS import CategoricalMDENAS
from xnas.search_algorithm.MIGO import MIGO
from xnas.search_algorithm.GridSearch import GridSearch
from xnas.search_algorithm.SNG import SNG, Dynamic_SNG
from sng_utils import EpochSumCategoryTestFunction, SumCategoryTestFunction


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
        return CategoricalDDPNAS(category, 100)
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
    test_fun = EpochSumCategoryTestFunction(
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
    for i in tqdm.tqdm(range(running_times)):
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
