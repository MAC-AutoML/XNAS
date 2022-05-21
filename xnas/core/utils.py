import decimal
import random
import string
import time
import numpy as np


def random_time_string(stringLength=8):
    letters = string.ascii_lowercase
    return str(time.time()).join(random.choice(letters) for i in range(stringLength))


def one_hot_to_index(one_hot_matrix):
    return np.array([np.where(r == 1)[0][0] for r in one_hot_matrix])


def index_to_one_hot(index_vector, C):
    return np.eye(C)[index_vector.reshape(-1)]


def float_to_decimal(data, prec=4):
    """Convert floats to decimals which allows for fixed width json."""
    if isinstance(data, dict):
        return {k: float_to_decimal(v, prec) for k, v in data.items()}
    if isinstance(data, float):
        return decimal.Decimal(("{:." + str(prec) + "f}").format(data))
    else:
        return data


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
