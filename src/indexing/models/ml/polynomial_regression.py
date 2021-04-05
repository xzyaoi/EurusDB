# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from timeit import default_timer as timer

import src.indexing.utilities.metrics as metrics
from src.indexing.learning.polynomial_regression import PolynomialRegression
from src.indexing.models import BaseModel


class PRModel(BaseModel):
    def __init__(self, degree, page_size) -> None:
        super().__init__("Polynomial Regression with degree {}".format(degree),
                         page_size)
        self.model = PolynomialRegression(degree)

    def train(self, x_train, y_train, x_test, y_test):
        start_time = timer()
        self.model.fit(x_train, y_train)
        end_time = timer()
        yhat = self.model.predict(x_test)
        mse = metrics.mean_squared_error(y_test, yhat)
        return mse, end_time - start_time

    def predict(self, key):
        return self.model.predict(key)
