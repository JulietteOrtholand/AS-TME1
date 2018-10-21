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
    
    def  backward_update_gradient(self, x, delta):
        ## Gradient du module par  rapport  aux parametres
        ## et  mise a jour du  gradient
        pass
    
    def update_parameters (self, epsilon):
        ## Mise a jour  des  parametres
        pass
    
    def backward_delta (self, x, delta):
        ## Retourne  l e  gradient  du module par  rapport  aux  entrees
        pass
    
    def zero_grad(self):
        ## Remise a zero du  gradient    
        pass
    
    def  initialize_parameters(self):
        ### Initialisation des parametres
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
    
    def update_parameters(self, epsilon):
        self.w = Variable(self.w - epsilon*self.w.grad, requires_grad=True)
    

#########################
### Module activation ###
#########################
class Activation(Module):
    def __init__(self, fonc):
        self.fonc = fonc
    
    def forward(self, x):
        return self.fonc(x)


