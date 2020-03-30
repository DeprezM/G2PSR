# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:01:39 2020

@author: morei
"""


import torch as pt
import numpy as np
from matplotlib import pyplot as plt

n=100
d=50
q=3
nsr=0.05

DEVICE = pt.device('cuda:' if pt.cuda.is_available() else 'cpu')
        
class autoencoder(pt.nn.Module):
    
    def __init__(self, input_shape, latentdim):
        super().__init__()
        self.W_mu=pt.nn.Linear(input_shape[1], latentdim)
        self.W_logvar=pt.nn.Linear(input_shape[1], latentdim)
        self.W_mu_out=pt.nn.Linear(latentdim, input_shape[1])
        self.W_logvar_out=pt.nn.Linear(latentdim, input_shape[1])
        self.criterion = pt.nn.MSELoss()
        self.optimizer = pt.optim.Adam(self.parameters())
    
    def encode(self, X):
        Z=[]
        for i, xi in enumerate(X):
            q = pt.distributions.Normal(
                loc=self.W_mu(xi),
                scale=self.W_logvar(xi).exp().pow(0.5)
			)
            Z.append(q)
            del q
        return Z

    def sample(self, Z):
        if self.training:
            return Z.rsample()
        else:
            return Z.loc

    def decode(self, Z2):
        X2 = pt.distributions.Normal(
            loc=self.W_mu,
            scale=self.W_logvar.exp().pow(0.5)
		)
        return X2
    
    def forward(self,X):
        Z=self.encode(X)
        Z2=self.sample(Z)
        X2=self.decode(Z2)
        return {"X": X, "Z": Z, "X'":X2}
    
    def optimize(self,X, epochmax):
        losslist=[]
        for epoch in range(0, epochmax):
            self.optimizer.zero_grad()
            pred=self.forward(X)['X']
            loss=self.criterion(X, pred)
            loss.backward()
            losslist.append(loss.item())
            self.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        return self.state_dict()
    
