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
        self.latentdim=latentdim
        self.alpha=pt.Tensor([0.5] * latentdim)
        self.alpha.requires_grad=True
        self.optimizer = pt.optim.Adam(self.parameters())
        self.optimizer.add_param_group({ "params":self.alpha})

    def encode(self, X):
        Z = pt.distributions.Normal(
            loc=self.W_mu(X),
            scale=(self.alpha + 2 * pt.log(self.W_mu(X)**2 + 1e-8)).exp().pow(0.5) #self.W_mu**2 + 1e-8 can be replaced by pt.abs(self.W_mu + 1e-8)
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
    
    def loss_function(self, fwd_return):
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        X = fwd_return['X']
        X2 = fwd_return["X'"]
  
        kl = 0
        ll = 0
  
  			# KL Divergence
        kl -= (k1 * pt.sigmoid(k2 + k3 * self.alpha) - 0.5 * pt.log1p(self.alpha.exp().pow(-1)) - k1).mean(0)
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
        alpha1=[]
        alpha2=[]
        for epoch in range(0, epochmax):
            if (self.alpha[0] != self.alpha[0]):
                print(losslist[len(losslist)-1])
                break
            self.optimizer.zero_grad()
            pred=self.forward(X)
            loss=self.loss_function(pred)['total'].mean()
            loss.backward(retain_graph=True)
            losslist.append(loss)
            alpha1.append(self.alpha[0].item())
            alpha2.append(self.alpha[1].item())
            self.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        fig2=plt.figure()
        plt.plot(alpha1, figure=fig2)
        plt.plot(alpha2, figure=fig2)
        print(pred["Z"].rsample())
        return self.state_dict()
    
class VDonWeightAE(pt.nn.Module):
    
    def __init__(self, input_shape, output_shape, mu0=0.5, alpha0=0.5):
        super().__init__()
        if type(input_shape) is pt.Tensor:
            input_shape=input_shape.shape
        if type(output_shape) is pt.Tensor:
            output_shape=output_shape.shape
        if type(input_shape) is pt.Size or type(input_shape) is list:
            inputdim=input_shape[1]
        else: return("error")
        if type(output_shape) is pt.Size or type(output_shape) is list:
            outputdim=output_shape[1]
        elif type(output_shape) is int:
            outputdim=output_shape
        else: return("error")
        self.mu=pt.nn.Parameter(pt.Tensor([[mu0] * outputdim] * inputdim), requires_grad=True)
        self.alpha=pt.nn.Parameter(pt.Tensor([[0.5]] * inputdim), requires_grad=True) #alpha is a log to avoid getting nan
        self.optimizer = pt.optim.Adam(self.parameters())
        
    def probalpha(self):
        alpha=self.alpha.exp()
        p=pt.mul(alpha, 1/(alpha+1))
        return p
        
    def encode(self, X):
        alpha=self.alpha.exp()
        for i in range(1,self.mu.shape[1]):
            alpha=pt.cat((alpha, self.alpha.exp()), dim=1)
        pW=pt.distributions.Normal(self.mu,(alpha + 2 * pt.log(self.mu**2 + 1e-8)).exp().pow(0.5))
        W=pW.rsample()
        Y=X@W
        return Y
    
    def loss_function(self, trueY, pred):
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        kl = (k1 * pt.sigmoid(k2 + k3 * self.alpha) - 0.5 * pt.log1p(self.alpha.exp().pow(-1)) - k1).mean()
        cost = ((pred - trueY)**2).pow(.5).mean()
        return (cost-kl)
    
    def optimize(self,X, Y, epochmax):
        losslist=[]
        for epoch in range(0, epochmax):
            self.optimizer.zero_grad()
            pred=self.encode(X)
            loss=self.loss_function(Y, pred)
            loss.backward(retain_graph=True)
            losslist.append(loss)
            self.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        print(pred)
        return self.state_dict()
        