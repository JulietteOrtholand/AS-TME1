

import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def one_vs_one(X, y, c1, c2, test_size=0.3, validation_size = 0.3):
    X = torch.cat((X[y[:,c1] == 1], X[y[:,c2] == 1]))
    y = torch.cat((y[y[:,c1] == 1][:,[c1,c2]], y[y[:,c2] == 1][:,[c1,c2]]))
    inds = list(range(len(X)))
    np.random.shuffle(inds)
    sep = int(test_size * len(X))
    sep2 = int(test_size * len(X)) + int(validation_size * len(X))
    X.requires_grad = True
    y.requires_grad = True
    return X[inds][:sep], X[inds][sep:sep2],X[inds][sep2:], y[inds][:sep], y[inds][sep:sep2],y[inds][sep2:]

def split_train_test(X, y, test_size=0.3, validation_size = 0.3):
    inds = list(range(len(X)))
    np.random.shuffle(inds)
    sep = int(test_size * len(X))
    sep2 = int(test_size * len(X)) + int(validation_size * len(X))
    X.requires_grad = True
    y.requires_grad = True
    return X[inds][:sep], X[inds][sep: sep2], X[inds][sep2:], y[inds][:sep], y[inds][sep: sep2], y[inds][sep2:]
