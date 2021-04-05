# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tinyml
from tinyml.layers import Linear, ReLu
from tinyml.learner import Learner
from tinyml.losses import mse_loss
from tinyml.net import Sequential
from tinyml.optims import SGDOptimizer

# set this to 0 will omit training loss in every epoch.
# set this to 1 will print training loss in every epoch.
tinyml.utilities.logger.VERBOSE = 0


class FullyConnectedNetwork():
    def __init__(self, num_neurons, activations, lr=0.01) -> None:
        self.num_fc_layers = len(num_neurons) - 1
        if not len(activations) == self.num_fc_layers:
            raise ValueError(
                "Activations must be attached to every fully connected layer!")
        self.num_neurons = num_neurons
        self.activations = activations
        self.model = Sequential([])
        self.lr = lr
        self._build()

    def _build(self):
        for idx in range(self.num_fc_layers):
            self.model.add(
                Linear('fc_{}'.format(idx), self.num_neurons[idx],
                       self.num_neurons[idx + 1]))
            if self.activations[idx] == 'relu':
                self.model.add(ReLu('relu_{}'.format(idx)))
        self.model.build_params()
        # self.model.summary()

    def fit(self, X, y, epochs=200, batch_size=100) -> None:
        self.learner = Learner(self.model, mse_loss, SGDOptimizer(lr=self.lr))
        self.model, _ = self.learner.fit(X,
                                         y,
                                         epochs=epochs,
                                         batch_size=batch_size)

    def predict(self, X):
        return self.learner.predict(X)
