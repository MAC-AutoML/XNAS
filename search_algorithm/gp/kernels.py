import torch
import numpy as np


class Kernel(torch.nn.Module):
    """Base kernel."""

    def __add__(self, other):
        """Sums two kernels together.

        Args:
            other (Kernel): Other kernel.

        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.add)

    def __mul__(self, other):
        """Multiplies two kernels together.

        Args:
            other (Kernel): Other kernel.

        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.mul)

    def __sub__(self, other):
        """Subtracts two kernels from each other.

        Args:
            other (Kernel): Other kernel.

        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.sub)

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.

        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.

        Returns:
            Covariance (Tensor).
        """
        raise NotImplementedError


class AggregateKernel(Kernel):
    """An aggregate kernel."""

    def __init__(self, first, second, op):
        """Constructs an AggregateKernel.

        Args:
            first (Kernel): First kernel.
            second (Kernel): Second kernel.
            op (Function): Operation to apply.
        """
        super(Kernel, self).__init__()
        self.first = first
        self.second = second
        self.op = op

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.

        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.

        Returns:
            Covariance (Tensor).
        """
        first = self.first(xi, xj, *args, **kwargs)
        second = self.second(xi, xj, *args, **kwargs)
        return self.op(first, second)


class RBFKernel(Kernel):
    """Radial-Basis Function Kernel."""

    def __init__(self, length_scale=None, sigma_s=None, eps=1e-6):
        """Constructs an RBFKernel.

        Args:
            length_scale (Tensor): Length scale.
            sigma_s (Tensor): Signal standard deviation.
            eps (float): Minimum bound for parameters.
        """
        super(Kernel, self).__init__()
        self.length_scale = torch.nn.Parameter(
            torch.randn(1) if length_scale is None else length_scale)
        self.sigma_s = torch.nn.Parameter(
            torch.randn(1) if sigma_s is None else sigma_s)
        self._eps = eps

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.

        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.

        Returns:
            Covariance (Tensor).
        """
        length_scale = (self.length_scale ** -2).clamp(self._eps, 1e5)
        var_s = (self.sigma_s ** 2).clamp(self._eps, 1e5)

        M = torch.eye(xi.shape[1]) * length_scale
        dist = mahalanobis_squared(xi, xj, M)
        return var_s * (-0.5 * dist).exp()


class WhiteNoiseKernel(Kernel):
    """White noise kernel."""

    def __init__(self, sigma_n=None, eps=1e-6):
        """Constructs a WhiteNoiseKernel.

        Args:
            sigma_n (Tensor): Noise standard deviation.
            eps (float): Minimum bound for parameters.
        """
        super(Kernel, self).__init__()
        self.sigma_n = torch.nn.Parameter(
            torch.randn(1) if sigma_n is None else sigma_n)
        self._eps = eps

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.

        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.

        Returns:
            Covariance (Tensor).
        """
        var_n = (self.sigma_n ** 2).clamp(self._eps, 1e5)
        return var_n


class SampleKernel(Kernel):
    """docstring for Kernel"""

    def __init__(self, intra_class_distance=1,
                 inter_class_distance=2,
                 set_list=[[3, 4, 5, 6], [0, 1, 2], [7]]):
        super(Kernel, self).__init__()
        self.intra_class_distance = intra_class_distance
        self.inter_class_distance = inter_class_distance
        self.set_list = set_list

    def vector_distance(self, reference, target):
        """define the distance between two samples

        Args:
            reference: a list with 28 indexs
            target: a list with 28 indexs
        Returns:
            an int scaler, the sampling distance between two vectors
        """
        assert len(reference) == len(target)

        _dist = list()
        for r, t in zip(reference, target):
            for set in self.set_list:
                if r in set and t in set:
                    if r == t:
                        _dist.append(0)
                    else:
                        _dist.append(self.intra_class_distance)
                    break
                elif (r in set and t not in set) or (r not in set and t in set):
                    _dist.append(self.inter_class_distance)
                    break

        return sum(_dist) / float(len(_dist))

    def set_distance(self, set_r, set_t):
        """Define a distance matrix between two sets

        Args:
            set_r: reference set with shape (Nx, dim)
            set_t: target set with shape (Ny, dim)

        Returns:
            A distance matrix with shape (Nx, Ny)
        """
        if isinstance(set_r, torch.Tensor):
            set_r = set_r.numpy().tolist()
        if isinstance(set_t, torch.Tensor):
            set_t = set_t.numpy().tolist()

        dist = np.zeros([len(set_r), len(set_t)], dtype=np.float32)
        for i, r in enumerate(set_r):
            for j, t in enumerate(set_t):
                dist[i][j] = float(self.vector_distance(r, t))
        dist_tesnor = torch.from_numpy(dist)
        dist_tensor_gaussain = Gaussain_kernel(dist_tesnor)
        return dist_tensor_gaussain

    def forward(self, set_r, set_t, *args, **kwargs):
        # calulate intra-set distance matrix for both reference and target
        #intra_r = self.set_distance(set_r, set_r)
        #intra_t = self.set_distance(set_t, set_t)

        # calculate inter-set distance matrix between reference and targec
        inter_d = self.set_distance(set_r, set_t)

        return inter_d


def mahalanobis_squared(xi, xj, VI=None):
    """Computes the pair-wise squared mahalanobis distance matrix as:

        (xi - xj)^T V^-1 (xi - xj)

    Args:
        xi (Tensor): xi input matrix.
        xj (Tensor): xj input matrix.
        VI (Tensor): The inverse of the covariance matrix, default: identity
            matrix.

    Returns:
        Weighted matrix of all pair-wise distances (Tensor).
    """
    if VI is None:
        xi_VI = xi
        xj_VI = xj
    else:
        xi_VI = xi.mm(VI)
        xj_VI = xj.mm(VI)

    D = (xi_VI * xi).sum(dim=-1).reshape(-1, 1) \
        + (xj_VI * xj).sum(dim=-1).reshape(1, -1) \
        - 2 * xi_VI.mm(xj.t())
    return D


def Gaussain_kernel(x, sigma = 1):
    return torch.exp(-1 * (x ** 2/(2* sigma**2)))
