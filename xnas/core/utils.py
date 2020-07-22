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
