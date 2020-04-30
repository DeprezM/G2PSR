# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:54 2020

@author: morei
"""


import autoencoder as a
import torch as pt

samplesize=50
noise=0.05
q=10
g=5

X=pt.rand(samplesize,q,requires_grad=True)

test=a.VDAutoencoder(X.shape, g, 0.5).to(a.DEVICE)

print(X)

test.optimize(X, 4000)