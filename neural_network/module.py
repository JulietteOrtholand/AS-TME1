# coding: utf-8

import torch
import numpy as np
from function import *


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
        self.w = torch.rand(_in, out)
        self.grad = torch.zeros(_in, out)
        
    def forward(self, x):
        return FunctionLinear.forward(x, self.w)  
    
    def backward_update_gradient(self, x, delta):
        self.grad = FunctionLinear.backward(delta, x, self.w)[1]
    
    def backward_delta(self, x, delta):
        return torch.mm(delta, self.w.t())
    
    def update_parameters(self, epsilon):
        self.w = self.w - epsilon*self.grad
    
    def zero_grad(self):
        self.grad.zero_()


#########################
### Module activation ###
#########################
class Activation(Module):
    def __init__(self, fonc, foncg):
        self.fonc = fonc
        self.foncg = foncg
    
    def forward(self, x):
        return self.fonc(x)
    
    def backward_delta(self, x, delta):
        return self.foncg(x)*delta


##############################
### Fonctions d'activation ###
##############################
tanh = lambda x: np.tanh(x)

tanh_g = lambda x: 1-np.tanh(x)**2

sigmoid = lambda x: 1 / (1 + torch.exp(-x))

sigmoid_g = lambda x: sigmoid(x)*(1-sigmoid(x))

softmax = lambda x: torch.exp(x) / torch.sum(torch.exp(x))

def softmax2(x):
    stable_x = x - torch.max(x) # pour la stabilité lorsque x >>> trop grand
    return softmax(stable_x)

softmax_g = lambda x: softmax2(x) * (1 - softmax2(x))

