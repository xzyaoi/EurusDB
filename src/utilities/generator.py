# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random

import numpy as np

FACTOR = 10


def get_data(distribution, size):
    data = []
    if distribution == "UNIFORM":
        data = random.sample(range(size * FACTOR), size)
    elif distribution == "BINOMIAL":
        data = np.random.binomial(size * FACTOR, 0.4, size)
    elif distribution == "POISSON":
        data = np.random.poisson(200, size)
    elif distribution == "EXPONENTIAL":
        data = np.random.exponential(150, size)
    elif distribution == "LOGNORMAL":
        data = np.random.lognormal(0, 2, size)
    else:
        data = np.random.normal(size, size * FACTOR, size)
    data.sort()
    data = data + abs(np.min(data))
    return data
