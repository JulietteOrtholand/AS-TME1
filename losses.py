
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Loss:
    def forward(self, y, ypred):
        ## Calcule l'erreur
        pass

    def backward(self, y, ypred):
        ## Gradient du cout
        pass


class MSE(Loss):
    # Cout MSE
    def forward(self, y, ypred):
        return 0.5 * (ypred - y).pow(2).sum()

    def backward(self, y, ypred):
        return ypred - y


class Hinge(Loss):
    # Cout Hinge loss
    def forward(self, y, ypred):
        return np.maximum(0, -y * ypred).sum()

    def backward(self, y, ypred):
        return -y


class CE(Loss):
    # Cross Entropy
    def forward(self, y, ypred):
        # print(( y*np.log(ypred) + (1-y)*np.log(1-ypred) ).sum())
        return (-1 * y * np.log(ypred) - (1 - y) * np.log(1 - ypred)).sum()

    def backward(self, y, ypred):
        return -1 * y / ypred + (1 - y) / (1 - ypred)

