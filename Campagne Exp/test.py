# coding: utf-8

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import Sequential

import matplotlib.pyplot as plt

from tqdm import tqdm


######## JEU DE DONNEES MNIST ########
#----------------------------------------------------------------------
def get_dataset(batch_size, path):
    ### Obtenir train_loader, val_loader
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))
                ])), 
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))
                ])), 
        batch_size=batch_size, shuffle=False) 

    return train_loader, test_loader



######## NEURAL NETWORK ########
#----------------------------------------------------------------------
class NeuralNetwork():
    ### Classe implémentant un réseau de neurones et les fonctions utiles
    
    def __init__(self, model, loss, optim):
        ### Initialisation
        self.model = model
        self.loss = loss
        self.optim = optim


    def forward(self, X):
        ### Forward
        return self.model(X)


    def train(self, train_loader):
        ### Train epoch
        costsT, scoresT = [], []

        self.model.train()
        for i, (_data, target) in enumerate(train_loader):

            # Un seul vecteur
            data = _data.view(_data.shape[0], -1)

            # Forward
            out = self.forward(data)
            cost = self.loss.forward(out, target)

            self.optim.zero_grad()
            cost.backward(retain_graph=True)
            self.optim.step()

            costsT.append( cost.item() )
            scoresT.append( self.accuracy(out, target) )

        return sum(costsT)/len(costsT), sum(scoresT)/len(scoresT)


    def test(self, test_loader):
        ### Test epoch
        costsT, scoresT = [], []

        self.model.eval()
        for i, (_data, target) in enumerate(test_loader):

            # Un seul vecteur
            data = _data.view(_data.shape[0], -1)

            # Forward
            out = self.forward(data)
            cost = self.loss.forward(out, target)

            costsT.append( cost.item() )
            scoresT.append( self.accuracy(out, target) )

        return sum(costsT)/len(costsT), sum(scoresT)/len(scoresT)


    def fit_eval(self, train_loader, test_loader, n_epochs=100):
        ### Fit and eval
        costsT, costsV = [], []
        scoresT, scoresV = [], []

        for epoch in tqdm(range(n_epochs)):
            # Train
            csT = self.train(train_loader)
            costsT.append( csT[0] )
            scoresT.append( csT[1] )

            # Test
            csV = self.test(test_loader)
            costsV.append( csV[0] )
            scoresV.append( csV[1] )

        return costsT, scoresT, costsV, scoresV


    def accuracy(self, ypred, y):
        ### Accuracy
        ### y n'est pas en one_hot
        score = 0
        for i in range(0,len(y)):
            if y[i] == ypred[i].argmax():
                score += 1
        return(score/len(y))



######## PLOT LOSS/ACCURACY (SCORE) ########
#----------------------------------------------------------------------
def plot_CS(costsT, scoresT, costsV, scoresV, title=""):
    ### Affichage des courbes de loss et scores, en train et test
    plt.title("Cost : " + title)
    plt.plot(costsT, c="red", label = 'train')
    plt.plot(costsV, c="purple", label = 'test')
    plt.legend(loc="best")
    plt.show()

    plt.title("Score : " + title)
    plt.plot(scoresT, c="red", label = 'train')
    plt.plot(scoresV, c="purple", label = 'test')
    plt.legend(loc="best")
    plt.show()
