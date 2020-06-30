import random
import string
import time


def random_time_string(stringLength=8):
    letters = string.ascii_lowercase
    return str(time.time()).join(random.choice(letters) for i in range(stringLength))
