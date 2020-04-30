# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:01:39 2020

@author: morei
"""


import torch as pt
import numpy as np
from matplotlib import pyplot as plt

DEVICE = pt.device('cuda:' if pt.cuda.is_available() else 'cpu')
        
class Autoencoder(pt.nn.Module):
    
    def __init__(self, input_shape, latentdim):
        super().__init__()
        if type(input_shape) is pt.Size:
            inputdim=input_shape[1]
        elif type(int(input_shape)) is int:
            inputdim=input_shape
        self.W_mu=pt.nn.Linear(inputdim, latentdim)
        self.W_logvar=pt.nn.Linear(inputdim, latentdim)
        self.W_mu_out=pt.nn.Linear(latentdim, inputdim)
        self.W_logvar_out=pt.nn.Linear(latentdim, inputdim)
        self.optimizer = pt.optim.Adam(self.parameters())

    def encode(self, X):
        Z = pt.distributions.Normal(
            loc=self.W_mu(X),
            scale=self.W_logvar(X).exp().pow(0.5) #<- here change variance for wmu * alpha²
		)
        return Z
        

    def sample(self, Z):
        if self.training:
            return Z.rsample()
        else:
            return Z.loc

    def decode(self, Z2):
        X2 = pt.distributions.Normal(
            loc=self.W_mu_out(Z2),
            scale=self.W_logvar_out(Z2).exp().pow(0.5)
		)
        return X2
    
    def forward(self,X):
        Z=self.encode(X)
        Z2=self.sample(Z)
        X2=self.decode(Z2)
        return {"X": X, "Z": Z, "X'":X2}
    
    def loss_function(fwd_return):
        X = fwd_return['X']
        Z = fwd_return['Z']
        X2 = fwd_return["X'"]
  
        kl = 0
        ll = 0
  
  			# KL Divergence
        kl += pt.distributions.kl_divergence(Z, pt.distributions.Normal(0, 1)).sum(1).mean(0)  # torch.Size([1])
        ll += X2.log_prob(X).sum(1).mean(0)
  
        total = kl - ll
  
        losses = {
  			'total': total,
  			'kl': kl,
  			'll': ll
  		}
        
        return losses

    
    def optimize(self,X, epochmax):
        losslist=[]
        for epoch in range(0, epochmax):
            self.optimizer.zero_grad()
            pred=self.forward(X)
            loss=Autoencoder.loss_function(pred)['total']
            loss.backward()
            losslist.append(loss.item())
            self.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        print(pred)
        return self.state_dict()
    
class Autoencoder_multiplelayers(pt.nn.Module):
    
    def __init__(self, input_shape, dimgene, dimpathways):
        super().__init__()
        self.outerAE=Autoencoder(input_shape, dimgene)
        self.innerAE=Autoencoder(dimgene, dimpathways)
        
        self.W_mu=pt.nn.Linear(input_shape[1], dimpathways)
        self.W_logvar=pt.nn.Linear(input_shape[1], dimpathways)
        self.W_mu_out=pt.nn.Linear(dimpathways, input_shape[1])
        self.W_logvar_out=pt.nn.Linear(dimpathways, input_shape[1])
        
        self.optimizer = pt.optim.Adam(self.parameters())
        
    def encode(self, X):
        Z = pt.distributions.Normal(
            loc=self.W_mu(X),
            scale=self.W_logvar(X).exp().pow(0.5)
		)
        return Z
        

    def sample(self, Z):
        if self.training:
            return Z.rsample()
        else:
            return Z.loc

    def decode(self, Z2):
        X2 = pt.distributions.Normal(
            loc=self.W_mu_out(Z2),
            scale=self.W_logvar_out(Z2).exp().pow(0.5)
		)
        return X2
    
    def forward(self,X):
        Z=self.encode(X)
        Z2=self.sample(Z)
        X2=self.decode(Z2)
        return {"X": X, "Z": Z, "X'":X2}
    
    def loss_function(fwd_return):
        X = fwd_return['X']
        Z = fwd_return['Z']
        X2 = fwd_return["X'"]
  
        kl = 0
        ll = 0
  
  			# KL Divergence
        kl += pt.distributions.kl_divergence(Z, pt.distributions.Normal(0, 1)).sum(1).mean(0)  # torch.Size([1])
        ll += X2.log_prob(X).sum(1).mean(0)
  
        total = kl - ll
  
        losses = {
  			'total': total,
  			'kl': kl,
  			'll': ll
  		}
        
        return losses

    
    def optimize(self,X, epochmax, mode=0):
        if mode==0:
            self.optimize(X, epochmax, mode=1)
            self.optimize(self.outerAE.encode(X).rsample(), epochmax, mode=2)
            currentAE=self
        elif mode==1:
            currentAE=self.outerAE
        elif mode==2:
            currentAE=self.innerAE
        else:
            print("wrong mode")
            return
        losslist=[]
        for epoch in range(0, epochmax):
            currentAE.optimizer.zero_grad()
            pred=currentAE.forward(X)
            loss=Autoencoder.loss_function(pred)['total']
            loss.backward(retain_graph=True)
            losslist.append(loss.item())
            currentAE.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        print(pred)
        return self.state_dict()
    
class VDAutoencoder(pt.nn.Module):
    
    def __init__(self, input_shape, latentdim, alpha):
        super().__init__()
        if type(input_shape) is pt.Size:
            inputdim=input_shape[1]
        elif type(int(input_shape)) is int:
            inputdim=input_shape
        self.W_mu=pt.nn.Linear(inputdim, latentdim)
        self.W_logvar=pt.nn.Linear(inputdim, latentdim)
        self.W_mu_out=pt.nn.Linear(latentdim, inputdim)
        self.W_logvar_out=pt.nn.Linear(latentdim, inputdim)
        self.optimizer = pt.optim.Adam(self.parameters())
        self.alpha=alpha
        self.log_alpha=(pt.ones(latentdim) * alpha).log()

    def encode(self, X):
        Z = pt.distributions.Normal(
            loc=self.W_mu(X),
            scale=self.log_alpha + 2 * pt.log(pt.abs(self.W_mu(X)) + 1e-8) #<- here change variance for wmu * alpha²
		)
        return Z
        

    def sample(self, Z):
        if self.training:
            return Z.rsample()
        else:
            return Z.loc

    def decode(self, Z2):
        X2 = pt.distributions.Normal(
            loc=self.W_mu_out(Z2),
            scale=self.W_logvar_out(Z2).exp().pow(0.5)
		)
        return X2
    
    def forward(self,X):
        Z=self.encode(X)
        Z2=self.sample(Z)
        X2=self.decode(Z2)
        return {"X": X, "Z": Z, "X'":X2}
    
    def loss_function(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
  
        kl = 0
        ll = 0
  
  			# KL Divergence
        kl += -(c1 * pt.sigmoid(c2 + c3 * self.log_alpha) - 0.5 * pt.log1p(pt.exp(-self.log_alpha)) - c1)
  
        total = kl - ll
  
        losses = {
  			'total': total,
  			'kl': kl,
  			'll': ll
  		}
        
        return losses

    
    def optimize(self,X, epochmax):
        losslist=[]
        for epoch in range(0, epochmax):
            self.optimizer.zero_grad()
            pred=self.forward(X)
            loss=Autoencoder.loss_function(pred)['total']
            loss.backward()
            losslist.append(loss.item())
            self.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        print(pred)
        return self.state_dict()