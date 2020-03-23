# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:01:39 2020

@author: morei
"""


import torch as pt
import numpy as np
import random

n=100
d=50
q=3
nsr=0.05

pt.manual_seed(0)
X=np.empty([n,d], dtype=float)

DEVICE = pt.device('cuda:' if pt.cuda.is_available() else 'cpu')

for i in range(0,n):
    for i2 in range(0,d):
        X[i,i2]=random.randint(0,2)
        
X=pt.from_numpy(X)
        
class autoencoder(pt.nn.Module):
    
    def __init__(self, input_shape, latentdim):
        super().__init__()
        self.encoder=pt.nn.Linear(input_shape[1], latentdim)
        self.decoder=pt.nn.Linear(latentdim, input_shape[1])
        self.criterion = pt.nn.MSELoss()
        self.optimizer = pt.optim.Adam(self.parameters())
    
    def encode(self, X):
        Y=self.encoder(X)
        return Y
    
    def decode(self, X):
        Y=self.decoder(X)
        return Y
    
    def forward(self,X):
        Y=self.encode(X)
        X2=self.decode(Y)
        return {"X": X, "Y": Y, "X'":X2}
    
    def optimize(self,X, epochmax):
        for epoch in range(0, epochmax):
            self.optimizer.zero_grad()
            pred=self.forward(X)['X']
            loss=self.criterion(X, pred)
            loss.backward()
            self.optimizer.step()
            if(epoch==epochmax-1):
                print(pred)
        return self.state_dict()