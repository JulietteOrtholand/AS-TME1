# coding: utf-8

import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from neural_network_pytorch_grad import NeuralNetwork
from module import *
from loss import *


########################
### Fonctions utiles ###
#########################
#
def one_vs_one(X, y, c1, c2, test_size=0.3):
    ### Transformation pour la classification binaire, y en one hot
    X = torch.cat((X[y[:,c1] == 1], X[y[:,c2] == 1]))
    y = torch.cat((y[y[:,c1] == 1][:,[c1,c2]], y[y[:,c2] == 1][:,[c1,c2]]))
    inds = list(range(len(X)))
    np.random.shuffle(inds)
    sep = int(test_size * len(X))
    return X[inds][:sep], X[inds][sep:], y[inds][:sep], y[inds][sep:]

def split_train_test(X, y, test_size=0.3):
    ### Split train et test
    inds = list(range(len(X)))
    np.random.shuffle(inds)
    sep = int(test_size * len(X))
    return X[inds][:sep], X[inds][sep:], y[inds][:sep], y[inds][sep:]

def plot_cost_score(costs, costs_test, scores, scores_test):
    ### Affichage cout et score
    plt.title("Coût - train")
    plt.plot(range(len(costs)), costs, c="red")
    plt.show()

    plt.title("Coût - test")
    plt.plot(range(len(costs_test)), costs_test, c="purple")
    plt.show()

    plt.title("Score - train, test")
    plt.plot(range(len(scores)), scores, c="blue", label="train")
    plt.plot(range(len(scores_test)), scores_test, c="green", label="test")
    plt.legend(loc="best")
    plt.show()

def main():
    ### Chargement des données ###

    ## une fois le dataset telecharge, mettre download=False !
    ## Pour le test, train = False
    ## transform permet de faire un preprocessing des donnees (ici ?)
    batch_size=2000
    nb_digits=10
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size, shuffle=True) 

    y_onehot = torch.FloatTensor(batch_size, nb_digits) 

    for i,(data,target) in enumerate(train_loader):
        #print(i,data.size(),data.type(),target.size(),
        #   target.type())
        # do something...
        X = data.view(batch_size, -1)

        ## Encoding des labels en onehot
        y_onehot.zero_()
        y_onehot.scatter_(1, target.view(-1,1), 1)
        break


    X = torch.cat( (X, torch.ones(X.shape[0], 1)), 1 )
    y = y_onehot

    X_train, X_test, y_train, y_test = one_vs_one(X, y, 5, 7)

    nn = NeuralNetwork(loss=MSE())
    nn.add_layer( [ModuleLinear(len(X[0]), 2), Activation(torch.sigmoid)])
    res = nn.fit_and_test(X_train, y_train, X_test, y_test, 
        mode='mini_batch', epsilon=1e-1)
    
    plot_cost_score(*res)

if __name__ == "__main__":
    main()