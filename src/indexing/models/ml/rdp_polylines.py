# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from timeit import default_timer as timer

from src.indexing.learning.polynomial_regression import PolynomialRegression
from src.indexing.learning.rdp_algorithm import RDPAlgorithm
from src.indexing.models import BaseModel


class RDPModel(BaseModel):
    def __init__(self, epsilon) -> None:
        super().__init__("RDPModel {}".format(epsilon))
        self.epsilon = epsilon
        self.intervals = []
        self.lrs = []
        self.coeffs = []

    def fit(self, x_train, y_train):
        rdpa = RDPAlgorithm(self.epsilon)
        self.intervals = rdpa.fit(x_train)
        # split trainning set so that each part contains partial data
        self.interval_indices = []
        self.coeffs = []
        for idx, each in enumerate(self.intervals[0:-1]):
            index = self.smallest_larger(x_train, self.intervals[idx], each)
            self.interval_indices.append(index)
        # train linear regression with multiple processor
        for idx, each in enumerate(self.interval_indices[0:-1]):
            x_train_partial = x_train[each, self.interval_indices[idx + 1]]
            y_train_partial = y_train[each, self.interval_indices[idx + 1]]
            lr = PolynomialRegression(1)
            lr.fit(x_train_partial, y_train_partial)
            self.lrs.append(lr)

    def train(self, x_train, y_train, x_test, y_test):
        start_time = timer()
        self.fit(x_train, y_train)
        end_time = timer()
        yhat = []
        for each in x_test:
            yhat.append(self.predict(each))
        return end_time - start_time

    def predict(self, x):
        # find the proper interval
        index_of_model = 0
        for idx, each in enumerate(self.intervals):
            if x > each:
                if x < self.intervals[idx + 1]:
                    index_of_model = idx
        # use the linear regression
        return self.lrs[index_of_model].predict(x)

    def smallest_larger(self, x_train, start_idx, value):
        idx = (x_train[start_idx:] - value).argmin()
        return idx
