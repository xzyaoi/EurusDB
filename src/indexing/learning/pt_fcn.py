# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
'''
Fully Connected Neural Network backed by Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential


class FullyConnectedNetwork():
    def __init__(self, num_neurons, activations, lr=0.01) -> None:
        self.num_fc_layers = len(num_neurons) - 1
        if not len(activations) == self.num_fc_layers:
            raise ValueError(
                "Activations must be attached to every fully connected layer!")
        self.num_neurons = num_neurons
        self.activations = activations
        self.model = None
        self.lr = lr
        self._build()

    def _build(self):
        modules = []
        for idx in range(self.num_fc_layers):
            modules.append(
                nn.Linear(self.num_neurons[idx], self.num_neurons[idx + 1]))
            if self.activations[idx] == 'relu':
                modules.append(nn.ReLU())
        self.model = Sequential(*modules)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def fit(self, X, y, epochs=200, batch_size=100) -> None:
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        y = y.reshape(-1, 1)
        for i in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = F.mse_loss(output, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        X = torch.from_numpy(X).float()
        return self.model.forward(X).detach().numpy()
