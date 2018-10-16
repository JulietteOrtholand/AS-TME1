
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Module:
    def forward(self, x):
        ## Calcule la sortie
        pass

    def backward_update_gradient(self, x, delta):
        ## Gradient du module par  rapport  aux parametres
        ## et  mise a jour du  gradient
        pass

    def update_parameters(self, epsilon):
        ## Mise a jour  des  parametres
        pass

    def backward_delta(self, x, delta):
        ## Retourne  l e  gradient  du module par  rapport  aux  entrees
        pass

    def zero_grad(self):
        ## Remise a zero du  gradient
        pass

    def initialize_parameters(self):
        ### Initialisation des parametres
        pass


class FunctionLinear:
    @staticmethod
    def forward(x, w):
        return torch.mm(x, w)

    @staticmethod
    def backward(delta, x, w):
        # to check formula
        grad_x = torch.mm(delta, w.t())
        grad_w = torch.mm(x.t(), delta)
        return grad_x, grad_w


class ModuleLinear(Module):
    def __init__(self, _in, out):
        # in : dimension de l'entrée
        # out : dimension de la sortie
        self.w = torch.rand(_in, out)
        self.grad = torch.zeros(_in, out)

    def forward(self, x):
        return FunctionLinear.forward(x, self.w)

    def backward_update_gradient(self, x, delta):
        self.grad += FunctionLinear.backward(delta, x, self.w)[1]

    def backward_delta(self, x, delta):
        return torch.mm(delta, self.w.t())

    def update_parameters(self, epsilon):
        self.w = self.w - epsilon * self.grad

    def zero_grad(self):
        self.grad.zero_()


class ActivationFunction(Module):
    def __init__(self, fonc, foncg):
        self.fonc = fonc
        self.foncg = foncg

    def forward(self, x):
        return self.fonc(x)

    def backward_delta(self, x, delta):
        return self.foncg(x) * delta

