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

    def forward(self, x):
        return FunctionLinear.forward(x, self.w)


#########################
### Module activation ###
#########################
class Activation(Module):
    def __init__(self, fonc):
        self.fonc = fonc
        self.w = None

    def forward(self, x):
        return self.fonc(x)

