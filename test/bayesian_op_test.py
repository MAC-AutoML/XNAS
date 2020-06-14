from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
from test.search_algorithm_test_function import EpochSumCategoryTestFunction
import tqdm
import time
import os
import pdb


def trans(sample, func, M):
    bias = 0
    scale = 1
    if func == 'rastrigin':
        # search range is [-5.12, 5.12]
        scale = 10. / float(M)
        bias = int(M / 2)

    elif func == 'rosenbrock':
        # search range is (-inf, +inf)
        scale = 1.
        bias = int(M / 2)
    return (sample - bias) * scale


M = 20
N = 20
category = [M]*N
# test_function = SumCategoryTestFunction(category)
# ['quad', 'linear', 'exp', 'constant']
# ['index_sum', 'rastrigin', 'rosenbrock ']
epoc_function = 'linear'
func = 'rastrigin'
test_function = EpochSumCategoryTestFunction(category, epoch_func=epoc_function, func=func)


def get_function_and_bound(N, M):
    # assert func in ['rastrigin', 'rosenbrock']
    assert N <= 20
    name_list = ['a', 'b', 'c', 'd', 'e',
                 'f', 'g', 'h', 'i', 'j',
                 'k', 'l', 'm', 'n', 'o',
                 'p', 'q', 'r', 's', 't']
    bound = {}
    this_name_list = name_list[0: N]
    def_str = "def f("
    for i in this_name_list:
        def_str += i + ','
    def_str += "):\n"
    def_str += "  input_ = np.zeros({}) \n".format(str(N))
    for i in range(N):
        bound[this_name_list[i]] = (0, M)
        def_str += "  input_[{}] = int({}) \n".format(str(i), this_name_list[i])
    def_str += '  return test_function.objective_function(input_)'
    return def_str, bound


def run(func, N, M):
    # M = 20
    # N = 20
    category = [M] * N
    # test_function = SumCategoryTestFunction(category)
    # ['quad', 'linear', 'exp', 'constant']
    # ['index_sum', 'rastrigin', 'rosenbrock ']
    epoc_function = 'linear'
    # func = 'rastrigin'
    test_function = EpochSumCategoryTestFunction(category, epoch_func=epoc_function, func=func)
    func_str, bound = get_function_and_bound(N, M)
    exec(func_str)
    # pdb.set_trace()
    test_function.re_new()
    runing_times = 20
    runing_epochs = 50
    save_dir = '/userhome/project/Auto_NAS_V2/experiments/toy_example/'
    file_name = '{}_{}_{}_{}_{}_{}.npz'.format('bayesian_op', str(N), str(M), str(runing_epochs),
                                               epoc_function, func)
    file_name = os.path.join(save_dir, file_name)
    l2_distance = np.zeros([runing_times, runing_epochs]) - 1
    running_interval = np.zeros([runing_times, runing_epochs])
    for run_ in tqdm.tqdm(range(runing_times)):
        optimizer = BayesianOptimization(
            f=None,
            pbounds=bound,
            verbose=2,
            random_state=1,
        )
        # using the package
        # optimizer.maximize(
        #     init_points=1,
        #     n_iter=runing_epochs,
        #     alpha=1e-3)
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        for _ in range(runing_epochs):
            print('epoch: ' + str(_))
            start_time = time.time()
            next_point = optimizer.suggest(utility)
            target = f(**next_point)
            optimizer.register(params=next_point, target=target)
            end_time = time.time()
            print(end_time - start_time)
            running_interval[run_, _] = end_time - start_time
        best = -np.inf
        best_sample = np.zeros(N)
        for i, res in enumerate(optimizer.res):
            if res['target'] > best:
                best = res['target']
                sample = np.zeros(N)
                for j, key in enumerate(res['params'].keys()):
                    sample[j] = res['params'][key]
                best_sample = sample
            distance = test_function.l2_distance(best_sample)
            l2_distance[run_, i] = distance
    np.savez(file_name, l2_distance, running_interval)


if __name__ == '__main__':
    for func in ['index_sum', 'rastrigin', 'rosenbrock']:
        for (N, M) in [(10, 10), (20, 20)]:
            print(func + str(N) + str(M))
            func_str, bound = get_function_and_bound(N, M)
            exec(func_str)
            run(func, N, M)