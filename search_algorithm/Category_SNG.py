#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from search_algorithm.Category_dist import Categorical
from utils import utils


class SNG:
    """
    Stochastic Natural Gradient for Categorical Distribution
    """
    def __init__(self, categories, delta_init=1., lam=2, init_theta=None, max_mize=True):

        # self.N = np.sum(categories - 1)
        # Categorical distribution
        self.p_model = Categorical(categories)
        # valid dimension size
        self.p_model.C = np.array(self.p_model.C)
        self.valid_d = len(self.p_model.C[self.p_model.C > 1])

        if init_theta is not None:
            self.p_model.theta = init_theta

        # Natural SG
        self.delta = delta_init
        self.lam = lam  # lambda_theta
        self.eps = self.delta
        self.sample = []
        self.objective = []
        self.maxmize = -1 if max_mize else 1

    def get_lam(self):
        return self.lam

    def get_delta(self):
        return self.delta

    def record_information(self, sample, objective):
        self.sample.append(sample)
        self.objective.append(objective*self.maxmize)

    def sampling(self):
        """
        Draw a sample from the categorical distribution (one-hot)
        """
        rand = np.random.rand(self.p_model.d, 1)  # range of random number is [0, 1)
        cum_theta = self.p_model.theta.cumsum(axis=1)  # (d, Cmax)

        # x[i, j] becomes 1 if cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        c = (cum_theta - self.p_model.theta <= rand) & (rand < cum_theta)
        return c

    def sampling_index(self):
        return utils.one_hot_to_index(np.array(self.sampling()))

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
        if len(self.sample) == self.lam:
            sample_array = []
            objective = np.array(self.objective)
            if len(np.array(self.sample).shape) == 2:
                for sample in self.sample:
                    sample_array.append(utils.index_to_one_hot(sample, self.p_model.Cmax))
            sample_array = np.array(sample_array)
            self.update_function(sample_array, objective)
            self.sample = []
            self.objective = []

    def update_function(self, c_one, fxc, range_restriction=True):

        aru, idx = self.utility(fxc)
        if np.all(aru == 0):
            # If all the points have the same f-value,
            # nothing happens for theta and breaks.
            # In this case, we skip the rest of the code.
            return

        ng = np.mean(aru[:, np.newaxis, np.newaxis] * (c_one[idx] - self.p_model.theta), axis=0)

        sl = []
        for i, K in enumerate(self.p_model.C):
            theta_i = self.p_model.theta[i, :K - 1]
            theta_K = self.p_model.theta[i, K - 1]
            s_i = 1. / np.sqrt(theta_i) * ng[i, :K - 1]
            s_i += np.sqrt(theta_i) * ng[i, :K - 1].sum() / (theta_K + np.sqrt(theta_K))
            sl += list(s_i)
        sl = np.array(sl)

        pnorm = np.sqrt(np.dot(sl, sl)) + 1e-8
        self.eps = self.delta / pnorm
        self.p_model.theta += self.eps * ng

        for i in range(self.p_model.d):
            ci = self.p_model.C[i]
            # Constraint for theta (minimum value of theta and sum of theta = 1.0)
            theta_min = 1. / (self.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.
            self.p_model.theta[i, :ci] = np.maximum(self.p_model.theta[i, :ci], theta_min)
            theta_sum = self.p_model.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.p_model.theta[i, :ci] -= (theta_sum - 1.) * (self.p_model.theta[i, :ci] - theta_min) / tmp
            # Ensure the summation to 1
            self.p_model.theta[i, :ci] /= self.p_model.theta[i, :ci].sum()

    @staticmethod
    def utility(f, rho=0.25, negative=True):
        """
        Ranking Based Utility Transformation

        w(f(x)) / lambda =
            1/mu  if rank(x) <= mu
            0     if mu < rank(x) < lambda - mu
            -1/mu if lambda - mu <= rank(x)

        where rank(x) is the number of at least equally good
        points, including it self.

        The number of good and bad points, mu, is ceil(lambda/4).
        That is,
            mu = 1 if lambda = 2
            mu = 1 if lambda = 4
            mu = 2 if lambda = 6, etc.

        If there exist tie points, the utility values are
        equally distributed for these points.
        """
        eps = 1e-14
        idx = np.argsort(f)
        lam = len(f)
        mu = int(np.ceil(lam * rho))
        _w = np.zeros(lam)
        _w[:mu] = 1 / mu
        _w[lam - mu:] = -1 / mu if negative else 0
        w = np.zeros(lam)
        istart = 0
        for i in range(f.shape[0] - 1):
            if f[idx[i + 1]] - f[idx[i]] < eps * f[idx[i]]:
                pass
            elif istart < i:
                w[istart:i + 1] = np.mean(_w[istart:i + 1])
                istart = i + 1
            else:
                w[i] = _w[i]
                istart = i + 1
        w[istart:] = np.mean(_w[istart:])
        return w, idx

    def log_header(self, theta_log=False):
        header_list = ['delta', 'eps', 'theta_converge']
        if theta_log:
            for i in range(self.p_model.d):
                header_list += ['theta%d_%d' % (i, j) for j in range(self.C[i])]
        return header_list

    def log(self, theta_log=False):
        log_list = [self.delta, self.eps, self.p_model.theta.max(axis=1).mean()]

        if theta_log:
            for i in range(self.p_model.d):
                log_list += ['%f' % self.p_model.theta[i, j] for j in range(self.C[i])]
        return log_list

    def load_theta_from_log(self, theta_log):
        self.p_model.theta = np.zeros((self.p_model.d, self.p_model.Cmax))
        k = 0
        for i in range(self.p_model.d):
            for j in range(self.p_model.C[i]):
                self.p_model.theta[i, j] = theta_log[k]
                k += 1
