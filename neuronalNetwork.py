
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


class NeuralNetwork():

    def __init__(self, loss):
        self.loss = loss
        self.layers = []

    def forward(self, X):
        _in, outs = X, [X]

        for layer in self.layers:
            outs.append(layer.forward(_in))
            _in = outs[-1]
        return outs

    def backward(self, X, y, epsilon):
        outs = self.forward(X)
        cost = self.loss.forward(y, outs[-1])
        delta = self.loss.backward(y, outs[-1])

        for i, layer in enumerate(self.layers[::-1]):
            layer.zero_grad()  # ???

            layer.backward_update_gradient(outs[-i - 2], delta)
            layer.update_parameters(epsilon=epsilon)
            delta = layer.backward_delta(outs[-i - 2], delta)
        return cost

    def fit(self, X, y, mode='batch', max_iter=100, epsilon=1e-2):
        costs = []
        if mode == 'batch':
            for it in range(max_iter):
                costs.append(self.backward(X, y, epsilon=epsilon))

        elif mode == 'stoch':
            for it in range(max_iter):
                inds = list(range(len(X)))
                np.random.shuffle(inds)
                for xi, yi in zip(X[inds], y[inds]):
                    xi, yi = xi.reshape(1, -1), yi.reshape(1, -1)
                    costs.append(self.backward(xi, yi, epsilon=epsilon))

        else:
            for it in range(max_iter):
                inds = list(range(len(X)))
                np.random.shuffle(inds)
                X, y = X[inds], y[inds]
                for i in range(0, len(X), batch_size):
                    xi, yi = X[i:i + batch_size], y[i:i + batch_size]
                    costs.append(self.backward(xi, yi, epsilon=epsilon))

        return costs

    def add_layer(self, layer):
        if layer == None:
            return False
        elif type(layer) == list:
            self.layers.extend(layer)
        else:
            self.layers.append(layer)

    def pop_layer(self, layer):
        self.layers.pop(-1)