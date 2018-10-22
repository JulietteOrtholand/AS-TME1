# coding: utf-8

import torch

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

relu = lambda x: torch.max(x, torch.zeros(r.shape))

relu_g = lambda x: (torch.sign(x)+1)/2