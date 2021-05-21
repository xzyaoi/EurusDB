# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from timeit import default_timer as timer

import numpy as np

import src.indexing.utilities.metrics as metrics
#from src.indexing.learning.fully_connected_network import FullyConnectedNetwork
from src.indexing.learning.pt_fcn import FullyConnectedNetwork
from src.indexing.models import BaseModel
from src.indexing.utilities.dataloaders import normalize


class FCNModel(BaseModel):
    def __init__(self,
                 layers=[1, 32, 1],
                 activations=['relu', 'relu'],
                 epochs=1000,
                 page_size=32,
                 lr=0.01) -> None:
        super().__init__('Fully Connected Neural Network', page_size)
        self.net = FullyConnectedNetwork(layers, activations, lr=lr)
        self.has_normalized = False
        self.epochs = epochs
        self.lr = lr

    def _normalize(self, x_train, y_train, x_test, y_test):
        if not self.has_normalized:
            self.max_x = np.max(x_train)
            self.min_x = np.min(x_train)
            self.max_y = np.max(y_train)
            self.min_y = np.min(y_train)
            x_train, y_train = normalize(x_train), normalize(y_train)

            if x_test is not None:
                x_test = (x_test - self.min_x) / (self.max_x - self.min_x + 1)
            if y_test is not None:
                y_test = (y_test - self.min_y) / (self.max_y - self.min_y + 1)
            return x_train, y_train, x_test, y_test
        else:
            print("has already normalized...")

    def train(self, x_train, y_train, x_test, y_test):

        x_train, y_train, x_test, y_test = self._normalize(
            x_train, y_train, x_test, y_test)
        start_time = timer()
        self.net.fit(x_train, y_train, epochs=self.epochs, batch_size=10)
        end_time = timer()

        y_hat = self.net.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_hat)
        return mse, end_time - start_time

    def fit(self, x_train, y_train):
        '''
        fit
        '''
        x_train, y_train, x_test, y_test = self._normalize(
            x_train, y_train, None, None)
        self.net.fit(x_train, y_train, epochs=self.epochs, batch_size=100)

    def predict(self, X):
        X = (X - self.min_x) / ((self.max_x - self.min_x) + 1)
        X = np.array(X)
        X = X.reshape((1))
        portion = self.net.predict(X)[0]
        # print(portion)
        if (portion == np.nan):
            print("portion {}, max y {}, min y {}".format(
                portion, self.max_y, self.min_y))
        position = int(portion * (self.max_y - self.min_y)) + self.min_y
        return position
