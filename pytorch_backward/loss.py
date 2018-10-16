# coding: utf-8

import torch

#################
### Squelette ###
#################
class Loss:
    def forward(self, y, ypred):
        ## Calcule l'erreur
        pass
    
    def backward(self, y, ypred):
        ## Gradient du cout
        pass

###########
### MSE ###
###########
class MSE(Loss):
    # Cout MSE
    def forward(self, y, ypred):
        val = 0.5*(ypred - y).pow(2).sum()
        val.backward(retain_graph=True)
        return val

    def backward(self, y, ypred):
        return ypred - y



##################
### Hinge Loss ###
##################
class Hinge(Loss):
    # Cout Hinge loss
    def forward(self, y, ypred):
        val = torch.max(torch.zeros(y.shape), -y*ypred).sum()
        val.backward(retain_graph=True)
        return val

    def backward(self, y, ypred):
        return (-y*ypred > 0).float() * -y.float()

#####################
### Cross Entropy ###
#####################
class CrossEntropy(Loss):
    def forward(self, y, ypred):
        val = -1*(y*torch.log(ypred) + (1-y)*torch.log(1-ypred)).sum()
        val.backward(retain_graph=True)
        return val
    
    def backward(self, y, ypred):
        return -1*(y/ypred - (1-y)/(1-ypred))