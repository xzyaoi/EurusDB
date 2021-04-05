# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
'''
Polynomial Regression
'''

import numpy as np


class PolynomialRegression():
    def __init__(self, degree) -> None:
        self.degree = degree
        self.coeffs = None

    def fit(self, X, y) -> None:
        '''
        The function fit tries to fit a polynomial relation between X and y.
        The degree of the polynomial relation is determined by self.degree.
        When degree == 1, it falls back to linear regression.
        Args:
            X: (N,D) matrix.
            y: (N,) vector.
        '''
        N = len(X)
        matX = [np.ones(N)]
        for power in range(self.degree):
            matX.append(np.power(X, power + 1))
        matX = np.column_stack(matX)
        A = np.linalg.pinv(matX.T @ matX)
        D = A @ matX.T
        self.coeffs = D @ y

    def predict(self, X):
        y_pred = 0
        for power in range(self.degree):
            y_pred += self.coeffs[power + 1] * np.power(X, power + 1)
        y_pred + self.coeffs[0]
        return y_pred.astype(int)
