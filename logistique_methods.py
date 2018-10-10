
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import losses as ls

def stochastique(X, Y, module, activ, loss=ls.MSE(), epsilon=0.005, max_iter=10):
    cout = []
    for i in range(max_iter):
        inds = list(range(len(X)))
        np.random.shuffle(inds)
        for x, y in zip(X[inds], Y[inds]):
            x, y = x.reshape(1, -1), y.reshape(1, -1)

            module.zero_grad()
            s = module.forward(x)
            ypred = activ.forward(s)

            delta = loss.backward(y, ypred)
            delta = activ.backward_delta(s, delta)

            module.backward_update_gradient(x, delta)
            module.update_parameters(epsilon=epsilon)

            cout.append(loss.forward(y, ypred))

    return cout


def batch(X, Y, module, activ, loss=ls.MSE(), epsilon=0.05, max_iter=100):
    cout = []
    for i in range(max_iter):
        module.zero_grad()
        s = module.forward(X)
        ypred = activ.forward(s)

        delta = loss.backward(Y, ypred)
        delta = activ.backward_delta(s, delta)

        module.backward_update_gradient(X, delta)
        module.update_parameters(epsilon=epsilon)

        cout.append(loss.forward(Y, ypred))
    return cout


def mini_batch(X, Y, module, activ, loss=ls.MSE(), n=50, epsilon=0.05, max_iter=100):
    cout = []
    for i in range(max_iter):
        inds = list(range(len(X)))[:n]
        np.random.shuffle(inds)

        module.zero_grad()
        s = module.forward(X[inds])
        ypred = activ.forward(s)

        delta = loss.backward(Y[inds], ypred)
        delta = activ.backward_delta(s, delta)

        module.backward_update_gradient(X[inds], delta)
        module.update_parameters(epsilon=epsilon)

        cout.append(loss.forward(Y[inds], ypred))

    return cout