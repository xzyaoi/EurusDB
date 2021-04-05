# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from pympler import asizeof


def mean_squared_error(yhat, y):
    if not isinstance(yhat, np.ndarray):
        yhat = np.array(yhat)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    return (np.square(yhat.reshape(-1) - y.reshape(-1))).mean()


def get_memory_size(obj):
    '''
    return the memory size in kilo bytes
    '''
    return asizeof.asizeof(obj) / 1024
