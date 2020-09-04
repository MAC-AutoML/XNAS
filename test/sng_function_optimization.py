import sys
sys.path.append('.')
import argparse
import os
import time
import numpy as np
import tqdm

from xnas.search_algorithm.ASNG import ASNG, Dynamic_ASNG
from xnas.search_algorithm.DDPNAS import CategoricalDDPNAS
from xnas.search_algorithm.MDENAS import CategoricalMDENAS
from xnas.search_algorithm.MIGO import MIGO
from xnas.search_algorithm.GridSearch import GridSearch
from xnas.search_algorithm.SNG import SNG, Dynamic_SNG
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
        return MIGO(categories=category, step=step, lam=6,
                    pruning=True, sample_with_prob=sample_with_prob,
                    utility_function='log', utility_function_hyper=utility_function_hyper,
                    momentum=True, gamma=gamma)
    elif name == 'GridSearch':
        return GridSearch(category)
    else:
        raise NotImplementedError


def run(M=10, N=10, func='rastrigin', optimizer_name='SNG', runing_times=500, runing_epochs=200,
        step=4, gamma=0.9, save_dir=None, noise=0.0, sample_with_prob=True, utility_function='log',
        utility_function_hyper=0.4):
    category = [M]*N
    # test_function = SumCategoryTestFunction(category)
    # ['quad', 'linear', 'exp', 'constant']
    # ['index_sum', 'rastrigin', 'rosenbrock ']
    epoc_function = 'linear'
    test_function = EpochSumCategoryTestFunction(category, epoch_func=epoc_function, func=func, noise_std=noise)

    # distribution_optimizer = Category_DDPNAS.CategoricalDDPNAS(category, 3)
    distribution_optimizer = get_optimizer(optimizer_name, category, step=step, gamma=gamma,
                                           sample_with_prob=sample_with_prob, utility_function=utility_function,
                                           utility_function_hyper=utility_function_hyper)
    file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'.format(optimizer_name, str(N), str(M), str(runing_epochs),
                                                                 epoc_function, func, str(step), str(gamma), str(noise),
                                                                 str(sample_with_prob), utility_function, str(utility_function_hyper))
    # else:
    #     file_name = '{}_{}_{}_{}_{}_{}_{}.npz'.format(optimizer_name, str(N), str(M), str(runing_epochs),
    #                                                   epoc_function, func, str(noise))
    file_name = os.path.join(save_dir, file_name)
    record = {
        'objective': np.zeros([runing_times, runing_epochs]) - 1,
        'l2_distance': np.zeros([runing_times, runing_epochs]) - 1,
    }
    last_l2_distance = np.zeros([runing_times])
    running_time_interval = np.zeros([runing_times, runing_epochs])
    distance = 100
    for i in tqdm.tqdm(range(runing_times)):
        for j in range(runing_epochs):
            start_time = time.time()
            if hasattr(distribution_optimizer, 'training_finish') or j == (runing_epochs - 1):
                last_l2_distance[i] = distance
            if hasattr(distribution_optimizer, 'training_finish'):
                if distribution_optimizer.training_finish:
                    break
            sample = distribution_optimizer.sampling()
            print(sample)
            objective = test_function.objective_function(sample)
            distribution_optimizer.record_information(sample, objective)
            distribution_optimizer.update()
            current_best = np.argmax(distribution_optimizer.p_model.theta, axis=1)
            distance = test_function.l2_distance(current_best)
            record['objective'][i, j] = objective
            record['l2_distance'][i, j] = distance
            end_time = time.time()
            running_time_interval[i, j] = end_time - start_time
        test_function.re_new()
        del distribution_optimizer
        distribution_optimizer = get_optimizer(optimizer_name, category, step=step, gamma=gamma,
                                               sample_with_prob=sample_with_prob, utility_function=utility_function,
                                               utility_function_hyper=utility_function_hyper)
    # mean_obj = np.mean(record['objective'], axis=0)
    # mean_distance = np.mean(record['l2_distance'], axis=0)
    np.savez(file_name, record['l2_distance'], running_time_interval)
    return distribution_optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="function dimension", type=int, default=10)
    parser.add_argument("--M", help="dicrete level", type=int, default=10)
    parser.add_argument(
        "--func", help="test functions in [rastrigin, index_sum, rosenbrock]", type=str, default='rastrigin')
    parser.add_argument("--optimizer", help="dicrete level", type=str, default='SNG')
    parser.add_argument("--step", help="pruning step", type=int, default=4)
    parser.add_argument("--gamma", help="gamma value", type=float, default=0.9)
    parser.add_argument("--noise", help="noise std", type=float, default=0.0)
    parser.add_argument("-uh", "--utility_function_hyper",
                        help="the factor of utility_function", type=float, default=0.4)
    parser.add_argument("-ut", "--utility_function_type", help="the type of utility_function", type=str, default='log')
    parser.add_argument("-sp", "--sample_with_prob",  action='store_true')

    args = parser.parse_args()
    func = args.func
    step = args.step
    gamma = args.gamma
    save_dir = '/userhome/project/XNAS/experiment/MIGO/test_function'
    optimizer_name = args.optimizer
    print("N={}, M={}, function={}, step={}, gamma={}, optimizer={}, noise_std={}, utility_function_hyper={}, utility_function_type={}, sample_with_prob={}".format(
        str(args.N), str(args.M), func, str(step), str(gamma), optimizer_name, str(args.noise), str(args.utility_function_hyper), args.utility_function_type, str(args.sample_with_prob)))
    run(M=args.M, N=args.N, func=func, optimizer_name=optimizer_name, runing_times=500, runing_epochs=1000,
        step=step, gamma=gamma, save_dir=save_dir, noise=args.noise, sample_with_prob=args.sample_with_prob,
        utility_function=args.utility_function_type, utility_function_hyper=args.utility_function_hyper)
    # for func in ['rastrigin']:
    #     for step in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #         for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #             run(M=10, N=10, func=func, optimizer_name='dynamic_SNG_V3', runing_times=500, runing_epochs=1000,
    #                 step=step, gamma=gamma)
    # for func in ['index_sum', 'rastrigin', 'rosenbrock']:
    #     for M, N in [(10, 10), (20, 20)]:
    #         for optimizer_name in ['SNG', 'ASNG', 'dynamic_SNG', 'dynamic_ASNG', 'dynamic_SNG_V3']:
    #             print(func + str([N, M]) + optimizer_name)
    #             run(M=M, N=N, func=func, optimizer_name=optimizer_name, runing_times=1000, runing_epochs=200, step=4,
    #                 gamma=0.9)
    # run(M=10, N=10, func='rastrigin', optimizer_name='SNG', runing_times=500, runing_epochs=200, step=4, gamma=0.9)
