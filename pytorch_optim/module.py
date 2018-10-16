# coding: utf-8

import torch
import numpy as np
from function import *
from torch.autograd import Variable

#################
### Squelette ###
#################
class Module:
    def forward(self, x):
        ## Calcule la sortie
        pass

#######################
### Module Linéaire ###
#######################
class ModuleLinear(Module):
    def __init__(self, _in, out):
        #in : dimension de l'entrée
        #out : dimension de la sortie
        self.w = torch.rand(_in, out, requires_grad=True)
        self.grad = torch.zeros(_in, out)

    def forward(self, x):
        return FunctionLinear.forward(x, self.w)


#########################
### Module activation ###
#########################
class Activation(Module):
    def __init__(self, fonc, foncg):
        self.fonc = fonc
        self.foncg = foncg
        self.w = None

    def forward(self, x):
        return self.fonc(x)

##############################
### Fonctions d'activation ###
##############################
tanh = lambda x: torch.tanh(x)

tanh_g = lambda x: 1-torch.tanh(x)**2

sigmoid = lambda x: 1 / (1 + torch.exp(-x))

sigmoid_g = lambda x: sigmoid(x)*(1-sigmoid(x))

softmax = lambda x: torch.exp(x) / torch.sum(torch.exp(x))

def softmax2(x):
    stable_x = x - torch.max(x) # pour la stabilité lorsque x >>> trop grand
    return softmax(stable_x)

softmax_g = lambda x: softmax2(x) * (1 - softmax2(x))

