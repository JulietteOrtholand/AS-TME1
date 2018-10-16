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

