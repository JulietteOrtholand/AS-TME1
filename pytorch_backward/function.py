# coding: utf-8

import torch

class FunctionLinear:
    @staticmethod
    def forward( x, w):
        return torch.mm(x,w)  
    
    @staticmethod
    def backward(delta, x, w):
        #to check formula
        grad_x = torch.mm(delta, w.t())
        grad_w = torch.mm(x.t(),delta)
        return grad_x,grad_w


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