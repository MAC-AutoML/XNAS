import numpy as np

from .benchmark_base import Function


def fun_target(bx, a, b, c, r, s, t):
    assert len(bx.shape) == 1
    assert bx.shape[0] == 2
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)
    assert isinstance(r, float)
    assert isinstance(s, float)
    assert isinstance(t, float)

    y = a * (bx[1] - b * bx[0]**2 + c * bx[0] - r)**2 + s * (1 - t) * np.cos(bx[0]) + s
    return y


class Branin(Function):
    def __init__(self,
        a=1.0,
        b=5.1 / (4.0 * np.pi**2),
        c=5 / np.pi,
        r=6.0,
        s=10.0,
        t=1 / (8 * np.pi)
    ):
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(c, float)
        assert isinstance(r, float)
        assert isinstance(s, float)
        assert isinstance(t, float)

        num_dim = 2
        bounds = np.array([
            [-5, 10],
            [0, 15],
        ])
        global_minimizers = np.array([
            [-np.pi, 12.275],
            [np.pi, 2.275],
            [9.42478, 2.475],
        ])
        global_minimum = 0.397887
        function = lambda bx: fun_target(bx, a, b, c, r, s, t)

        Function.__init__(self, num_dim, bounds, global_minimizers, global_minimum, function)

