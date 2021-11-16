import numpy as np
from xnas.search_algorithm.utils import Categorical
from xnas.core.utils import index_to_one_hot, one_hot_to_index
import copy


class ASNG:
    """
    Adaptive Stochastic Natural Gradient for Categorical Distribution
    """

    def __init__(self, categories,
                 alpha=1.5, delta_init=1., lam=6,
                 Delta_max=np.inf, init_theta=None, max_mize=True):

        self.N = np.sum(np.array(categories) - 1)
        # Categorical distribution
        self.p_model = Categorical(categories)
        # valid dimension size
        self.p_model.C = np.array(self.p_model.C)
        self.valid_d = len(self.p_model.C[self.p_model.C > 1])

        if init_theta is not None:
            self.p_model.theta = init_theta

            # Adaptive SG
        self.alpha = alpha  # threshold for adaptation
        self.delta_init = delta_init
        self.lam = lam  # lambda_theta
        self.Delta_max = Delta_max  # maximum Delta (can be np.inf)

        self.Delta = 1.
        self.gamma = 0.0  # correction factor
        self.s = np.zeros(self.N)  # averaged stochastic natural gradient
        self.delta = self.delta_init / self.Delta
        self.eps = self.delta

        self.sample = []
        self.objective = []
        self.max_mize = -1 if max_mize else 1

    def get_lam(self):
        return self.lam

    def get_delta(self):
        return self.delta

    def record_information(self, sample, objective):
        self.sample.append(sample)
        self.objective.append(objective*self.max_mize)

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
        if len(self.sample) == self.lam:
            objective = np.array(self.objective)
            sample_array = np.array(self.sample)
            self.update_function(sample_array, objective)
            self.sample = []
            self.objective = []

    def update_function(self, c_one, fxc, range_restriction=True):
        self.delta = self.delta_init / self.Delta
        beta = self.delta / (self.N ** 0.5)

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
        # test code
        # sl_2 = []
        # for i, K in enumerate(self.p_model.C):
        #     theta_i = self.p_model.theta[i, :K]
        #     s_i = 1. / np.sqrt(theta_i) * ng[i, :K]
        #     sl_2 += list(s_i)
        # sl_2 = np.array(sl_2)
        # pnorm_2 = np.sqrt(np.dot(sl_2, sl_2)) + 1e-9

        pnorm = np.sqrt(np.dot(sl, sl)) + 1e-9
        self.eps = self.delta / pnorm
        self.p_model.theta += self.eps * ng

        self.s = (1 - beta) * self.s + np.sqrt(beta * (2 - beta)) * sl / pnorm
        self.gamma = (1 - beta) ** 2 * self.gamma + beta * (2 - beta)
        self.Delta *= np.exp(beta * (self.gamma - np.dot(self.s, self.s) / self.alpha))
        self.Delta = min(self.Delta, self.Delta_max)

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


# ASNG + DDPNAS
class Dynamic_ASNG:
    """
    Adaptive Stochastic Natural Gradient for Categorical Distribution
    """

    def __init__(self, categories,
                 alpha=1.5, delta_init=1., lam=6, step=3, pruning=True,
                 Delta_max=np.inf, init_theta=None, max_mize=True, sample_with_prob=True):

        self.N = np.sum(np.array(categories) - 1)
        # Categorical distribution
        self.p_model = Categorical(categories)
        # valid dimension size
        self.p_model.C = np.array(self.p_model.C)
        self.valid_d = len(self.p_model.C[self.p_model.C > 1])

        if init_theta is not None:
            self.p_model.theta = init_theta

            # Adaptive SG
        self.alpha = alpha  # threshold for adaptation
        self.delta_init = delta_init
        self.lam = lam  # lambda_theta
        self.Delta_max = Delta_max  # maximum Delta (can be np.inf)

        self.Delta = 1.
        self.gamma = 0.0  # correction factor
        self.s = np.zeros(self.N)  # averaged stochastic natural gradient
        self.delta = self.delta_init / self.Delta
        self.eps = self.delta

        self.sample = []
        self.objective = []
        self.max_mize = -1 if max_mize else 1

        # this is for dynamic distribution
        self.sample_with_prob = sample_with_prob
        self.ignore_index = []
        self.sample_index = []
        self.pruned_index = []
        self.pruning = pruning
        self.steps = step
        self.current_step = 1
        self.training_finish = False
        self.init_record()

    def init_record(self):
        for i in range(self.p_model.d):
            self.ignore_index.append([])
            self.sample_index.append(list(range(self.p_model.Cmax)))
            self.pruned_index.append([])

    def get_lam(self):
        return self.lam

    def get_delta(self):
        return self.delta

    def record_information(self, sample, objective):
        self.sample.append(sample)
        self.objective.append(objective*self.max_mize)

    def sampling(self):
        return index_to_one_hot(self.sampling_index(), self.p_model.Cmax)

    def sampling_index(self):
        # fairness sampling
        sample = []
        for i in range(self.p_model.d):
            # get the prob
            if self.sample_with_prob:
                prob = copy.deepcopy(self.p_model.theta[i, self.sample_index[i]])
                prob = prob / prob.sum()
                sample.append(np.random.choice(self.sample_index[i], p=prob))
            else:
                sample.append(np.random.choice(self.sample_index[i]))
            if len(self.sample_index[i]) > 0:
                self.sample_index[i].remove(sample[i])
        return np.array(sample)

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
        if len(self.sample_index[0]) == 0:
            objective = np.array(self.objective)
            sample_array = np.array(self.sample)
            self.update_function(sample_array, objective)
            self.sample = []
            self.objective = []
            self.current_step += 1
            if self.pruning and self.current_step > self.steps:
                # pruning the index
                pruned_weight = copy.deepcopy(self.p_model.theta)
                for index in range(self.p_model.d):
                    if not len(self.pruned_index[index]) == 0:
                        pruned_weight[index, self.pruned_index[index]] = np.nan
                    self.pruned_index[index].append(np.nanargmin(pruned_weight[index, :]))
                if len(self.pruned_index[0]) >= (self.p_model.Cmax - 1):
                    self.training_finish = True
                self.current_step = 1
            self.update_sample_index()

    def update_sample_index(self):
        for i in range(self.p_model.d):
            self.sample_index[i] = list(set(range(self.p_model.Cmax)) - set(self.pruned_index[i]))

    def update_function(self, c_one, fxc, range_restriction=True):
        self.delta = self.delta_init / self.Delta
        beta = self.delta / (self.N ** 0.5)

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

        pnorm = np.sqrt(np.dot(sl, sl)) + 1e-9
        self.eps = self.delta / pnorm
        self.p_model.theta += self.eps * ng

        self.s = (1 - beta) * self.s + np.sqrt(beta * (2 - beta)) * sl / pnorm
        self.gamma = (1 - beta) ** 2 * self.gamma + beta * (2 - beta)
        self.Delta *= np.exp(beta * (self.gamma - np.dot(self.s, self.s) / self.alpha))
        self.Delta = min(self.Delta, self.Delta_max)

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
