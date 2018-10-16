# coding: utf-8

import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from module import *
from loss import *

##############################
### Chargement des données ###
##############################

## une fois le dataset telecharge, mettre download=False !
## Pour le test, train = False
## transform permet de faire un preprocessing des donnees (ici ?)
batch_size=60000
nb_digits=10
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size, shuffle=True) 

y_onehot = torch.FloatTensor(batch_size, nb_digits) 

for i,(data,target) in enumerate(train_loader):
    #print(i,data.size(),data.type(),target.size(),
    #   target.type())
    # do something...
    X = data
    print(X.shape)


    ## Encoding des labels en onehot
    y_onehot.zero_()
    y_onehot.scatter_(1, target.view(-1,1), 1)

    print(y_onehot.shape)
    break

X = X.view(batch_size,28*28)
y = y_onehot


def one_vs_one(X, y, c1, c2, test_size=0.3):
    X = torch.cat((X[y[:,c1] == 1], X[y[:,c2] == 1]))
    y = torch.cat((y[y[:,c1] == 1], y[y[:,c2] == 1]))
    inds = list(range(len(X)))
    np.random.shuffle(inds)
    sep = int(test_size * len(X))
    return X[inds][:sep], X[inds][sep:], y[inds][:sep], y[inds][sep:]

X_train, X_test, y_train, y_test = one_vs_one(X, y, 3, 5)

nn = NeuralNetwork(loss=MSE())
nn.add_layer( [ModuleLinear(len(X[0]), 10), Activation(sigmoid, sigmoid_g)])
costs = nn.fit(X_train, y_train, mode='mini_batch')

plt.plot(range(len(costs)), costs)
plt.show()