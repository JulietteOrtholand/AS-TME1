# coding: utf-8

import torch

#################
### Squelette ###
#################
class Loss:
    def forward(self, y, ypred):
        ## Calcule l'erreur
        pass


###########
### MSE ###
###########
class MSE(Loss):
    # Cout MSE
    def forward(self, y, ypred):
        val = 0.5*(ypred - y).pow(2).sum()
        return val



##################
### Hinge Loss ###
##################
class Hinge(Loss):
    # Cout Hinge loss
    def forward(self, y, ypred):
        val = torch.max(torch.zeros(y.shape), -y*ypred).sum()
        return val

#####################
### Cross Entropy ###
#####################
class CrossEntropy(Loss):
    def forward(self, y, ypred):
        val = -1*(y*torch.log(ypred) + (1-y)*torch.log(1-ypred)).sum()
        return val
