# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib.pyplot import xlabel
from numpy.core.fromnumeric import mean

from src.indexing.utilities.metrics import mean_squared_error


class PiecewiseRegression():
    '''
    This class describes a regression model by piecewise linear function
    '''
    def __init__(self,
                 num_of_breakpoints=5,
                 stop_threshold=1e-5,
                 max_epochs=20000,
                 lrs=[0.01]) -> None:
        self.num_of_breakpoints = num_of_breakpoints
        self.stop_threshold = stop_threshold
        self.max_epochs = max_epochs
        self.lrs = lrs

    '''
    utilities
    '''

    def calculate_error(self, X, y, alphas, betas):
        _, A = self._calculate_alphas(betas, X, y)
        yhat = A @ alphas
        mse = mean_squared_error(yhat, y)
        return mse

    def _least_square(self, A, y):
        return np.linalg.pinv(A.T @ A) @ A.T @ y

    def _initialize_betas(self, X):
        indices = np.linspace(1,
                              len(X) - 1,
                              self.num_of_breakpoints).astype(np.uint8)
        betas = X[indices]
        betas = np.insert(betas, 0, X[0])
        return betas

    def _construct_A(self, betas, X):
        A = [np.ones_like(X)]
        A.append(X - betas[0])
        num_of_segments = len(betas) - 1
        for i in range(num_of_segments):
            A.append(np.where(X >= betas[i + 1], X - betas[i + 1], 0))
        A = np.vstack(A).T
        return A

    def _calculate_alphas(self, betas, X, y):
        A = self._construct_A(betas, X)
        alphas = self._least_square(A, y)
        return alphas, A

    def _calculate_gradient(self, A, alphas, betas, X, y):
        r = A @ alphas - y
        K = np.diag(alphas)
        G = [-1 * np.ones_like(X)]
        for i in range(len(betas)):
            G.append(np.where(X >= betas[i], -1, 0))
        G = np.vstack(G)
        return 2 * (K @ G @ r), 2 * (K @ G @ G.T @ K.T)

    def fit(self, X, y):
        '''
        This function tries to fit a piecewise linear function.
        '''
        betas = self._initialize_betas(X)
        alphas = None
        previous_error = 99999
        for i in range(self.max_epochs):
            alphas, A = self._calculate_alphas(betas, X, y)
            first_grad, second_grad = self._calculate_gradient(
                A, alphas, betas, X, y)
            s = -np.linalg.pinv(second_grad) @ first_grad
            # TODO: this needs to be double confirmed.
            # The idea is that we do not optimize the first beta
            s = s[1:]
            # Finds the best learning rate
            err = 99999
            for lr in self.lrs:
                trial_betas = betas + lr * s
                mse = self.calculate_error(X, y, alphas, trial_betas)
                if (mse <= err):
                    betas = trial_betas
                    err = mse
            if (np.abs(previous_error - err) <= self.stop_threshold):
                print('[info]: Early Stopping...')
                break
            previous_error = err
        self.alphas = alphas
        self.betas = betas
        return previous_error

    def predict(self, x):
        if not type(x) is np.ndarray:
            A = self._construct_A(self.betas, np.array([x]))
        else:
            A = self._construct_A(self.betas, x)
        yhat = A @ self.alphas
        return yhat
