# coding: utf-8

######################
### Neural Network ###
######################

import numpy as np
import torch
from torch.autograd import Variable

class NeuralNetwork():
    
    def __init__(self, loss):
        # Initialisation
        self.loss = loss
        self.layers = []
        self.param = []


    def forward(self, X):
        ### Forward
        _in, outs = X, [X]
        
        for layer in self.layers:
            outs.append( layer.forward(_in) )
            _in = outs[-1]
        return outs
    

    def backward(self, X, y, epsilon):
        self.optimize.zero_grad()
        costf = self.loss.forward(y, self.predict(X))
        costf.backward(retain_graph=True)
        self.optimize.step()
        return costf
    

    def fit_and_test(self, X_train, y_train, X_test, y_test, mode='batch', max_iter=100, epsilon=1e-2, batch_size=60):
        ### Entraine le modèle avec backpropagation et rend l'évolution du cout
        costs, costs_test,scores,scores_test = [], [],[],[]
        self.optimize = torch.optim.Adam(self.param)
        if mode=='batch':
            for it in range(max_iter):
                costs.append( self.backward(X_train, y_train, epsilon=epsilon) )
                costs_test.append( self.loss.forward(y_test, self.predict(X_test)) )
                scores_test.append(self.score(X_test,y_test))
                scores.append(self.score(X_train, y_train))

        elif mode=='stoch':
            for it in range(max_iter):
                inds = list(range(len(X_train)))
                np.random.shuffle(inds)
                for xi, yi in zip(X_train[inds], y_train[inds]):
                    xi, yi = xi.reshape(1,-1), yi.reshape(1,-1)
                    costs.append(self. backward(xi, yi, epsilon=epsilon) )
                    costs_test.append( self.loss.forward(y_test, self.predict(X_test)) )
                scores_test.append(self.score(X_test, y_test))
                scores.append(self.score(X_train, y_train))
        else:
            for it in range(max_iter):
                inds = list(range(len(X_train)))
                np.random.shuffle(inds)
                X_train, y_train = X_train[inds], y_train[inds]
                for i in range(0, len(X_train), batch_size):
                    xi, yi = X_train[i:i+batch_size], y_train[i:i+batch_size]
                    costs.append( self.backward(xi, yi, epsilon=epsilon) )
                    costs_test.append( self.loss.forward(y_test, self.predict(X_test)) )
                    scores_test.append(self.score(X_test, y_test))
                    scores.append(self.score(X_train,y_train))
        return  costs, costs_test,scores,scores_test

    def predict(self, X):
           # Prédiction
           return self.forward(X)[-1]


    def add_layer(self, layer):
        # Ajouter un module ou une liste de modules
        if layer==None:
            return False
        elif type(layer) == list:
            self.layers.extend( layer )
            for lala in layer:
                self.param.append({'params': lala.parameters()})
        else:
            self.layers.append( layer )
            self.param.append({'params': layer.parameters()})

    def score(self,X,y):
        ypred = self.forward(X)[-1]
        score = 0
        for i in range(0,len(y)):
            if y[i].argmax() == ypred[i].argmax():
                score += 1
        return(score/len(y))

    def pop_layer(self, layer):
        # Enlever le dernier module
        self.layers.pop(-1)

'''    def fit(self, X, y, mode='batch', max_iter=100, epsilon=1e-2, batch_size=60):
        ### Entraine le modèle avec backpropagation et rend l'évolution du cout
        costs = []

        if mode=='batch':
            for it in range(max_iter):
                costs.append( self.backward(X, y, epsilon=epsilon) )
            
        elif mode=='stoch':
            for it in range(max_iter):
                inds = list(range(len(X)))
                np.random.shuffle(inds)
                for xi, yi in zip(X[inds], y[inds]):
                    xi, yi = xi.reshape(1,-1), yi.reshape(1,-1)
                    costs.append(self. backward(xi, yi, epsilon=epsilon) )

        else:
            for it in range(max_iter):
                inds = list(range(len(X)))
                np.random.shuffle(inds)
                X, y = X[inds], y[inds]
                for i in range(0, len(X), batch_size):
                    xi, yi = X[i:i+batch_size], y[i:i+batch_size]
                    costs.append( self.backward(xi, yi, epsilon=epsilon) )
        return costs'''


