import numpy as np

EPSILON = 1e-6


class Function(object):
    def __init__(self, dimensionality, bounds, global_minimizers, global_minimum, function, dim_problem=None):
        assert isinstance(dimensionality, int) or dimensionality is np.inf
        assert isinstance(bounds, np.ndarray)
        assert isinstance(global_minimizers, np.ndarray)
        assert isinstance(global_minimum, float)
        assert callable(function)
        assert isinstance(dim_problem, int) or dim_problem is None

        self._dimensionality = dimensionality
        self._bounds = bounds
        self._global_minimizers = global_minimizers
        self._global_minimum = global_minimum
        self._function = function

        self.dim_problem = dim_problem

        self.validate_properties()

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def bounds(self):
        return self._bounds

    @property
    def global_minimizers(self):
        return self._global_minimizers

    @property
    def global_minimum(self):
        return self._global_minimum

    def function(self, bx):
        if self.dimensionality is np.inf:
            assert self.dim_problem is bx.shape[0]
        else:
            assert self.dimensionality is bx.shape[0]

        return self._function(bx)

    def output(self, X):
        assert isinstance(X, np.ndarray)

        if len(X.shape) == 2:
            list_results = [self.function(bx) for bx in X]
        else:
            list_results = [self.function(X)]

        by = np.array(list_results)
        Y = np.expand_dims(by, axis=1)

        return Y

    def validate_properties(self):
        shape_bounds = self.bounds.shape
        shape_global_minimizers = self.global_minimizers.shape

        assert len(shape_bounds) == 2
        assert shape_bounds[1] == 2
        assert len(shape_global_minimizers) == 2

        assert np.all((self.output(self.global_minimizers) - self.global_minimum) < EPSILON)

        if self.dimensionality is np.inf:
            pass
        else:
            assert self.dimensionality == shape_bounds[0] == shape_global_minimizers[1]

