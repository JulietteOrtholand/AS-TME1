import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import losses as ls


def stochastique(X, Y, module, loss=ls.MSE(), epsilon=0.0001, max_iter=100):
    cout = []
    for i in range(max_iter):
        inds = list(range(len(X)))
        np.random.shuffle(inds)
        for x, y in zip(X[inds], Y[inds]):
            x, y = x.reshape(1, -1), y.reshape(1, -1)

            module.zero_grad()
            ypred = module.forward(x)
            delta = loss.backward(y, ypred)

            module.backward_update_gradient(x, delta)
            module.update_parameters(epsilon=epsilon)

            cout.append(loss.forward(y, ypred))
    return cout

def batch(X, Y, module, loss=ls.MSE(), epsilon=0.005, max_iter=100):
    cout = []
    for i in range(max_iter):
        module.zero_grad()
        ypred = module.forward(X)
        delta = loss.backward(Y, ypred)

        module.backward_update_gradient(X, delta)
        module.update_parameters(epsilon=epsilon)

        cout.append(loss.forward(Y, ypred))
    return cout


def mini_batch(X, Y, module, loss=ls.MSE(), n=50, epsilon=0.05, max_iter=100):
    cout = []
    for i in range(max_iter):
        inds = list(range(len(X)))[:n]
        np.random.shuffle(inds)

        module.zero_grad()

        ypred = module.forward(X[inds])
        delta = loss.backward(Y[inds], ypred)

        module.backward_update_gradient(X[inds], delta)
        module.update_parameters(epsilon=epsilon)

        cout.append(loss.forward(Y[inds], ypred))

    return cout