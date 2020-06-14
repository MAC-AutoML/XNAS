from search_algorithm.gp.kernels import SampleKernel
from search_algorithm.gp.gp import GaussianProcess
from random import choices
import numpy as np
import torch
import itertools
from numpy import linalg as LA
from sklearn.metrics import pairwise_distances


def secret_function(x, noise=0.0):
    x = torch.sum(x, dim=1).reshape(-1,1)
    return x + noise * torch.randn(x.shape)


def kernel_test():
    kernel = SampleKernel(intra_class_distance=1, inter_class_distance=2)
    data_size = 100
    sample_list = [0, 1, 2, 3, 4, 5, 6, 7]
    X_ = [choices(sample_list, k = 14)for i in range(100)]
    X_ = np.array(X_)
    distance = kernel(X_, X_)

def gp_test():
    data_size = 10
    sample_list = [0, 1, 2, 3, 4, 5, 6, 7]
    X_ = [choices(sample_list, k=14) for i in range(data_size)]
    X = torch.Tensor(np.array(X_))
    Y = secret_function(X, noise=1e-1)
    x_test = [choices(sample_list, k=14) for i in range(data_size)]
    x_test = torch.Tensor(np.array(x_test))
    y_test = secret_function(x_test, noise=1e-1)
    kernel = SampleKernel(intra_class_distance=1, inter_class_distance=2)
    gp = GaussianProcess(kernel)
    gp.set_data(X, Y)
    gp.fit()
    z_test, std_dev = gp(x_test, return_std=True)
    pass


if __name__ == '__main__':
    # gp_test()
    # decompose test
    D = np.zeros([8, 8])
    intra_class_distance = 1
    inter_class_distance = 2
    classes = [[0, 1, 2, 3], [4, 5, 6], [7]]
    for (i, j) in itertools.product(range(8),range(8)):
        if i == j:
            continue
        for set in classes:
            if i in set and j in set:
                D[i, j] = intra_class_distance
                break
            elif (i in set and j not in set) or (i not in set and j in set):
                D[i, j] = inter_class_distance
                break
    M = np.zeros([8, 8])
    for (i, j) in itertools.product(range(8), range(8)):
        M[i, j] = (D[i, 0]**2 + D[0, j] ** 2 - D[i, j]**2) / 2.
    w, v = LA.eig(M)
    X = v.dot(np.sqrt(np.diag(w)))
    D_new = pairwise_distances(X, X)
    pass