# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:54 2020

@author: morei
"""


import autoencoder as a
import torch as pt

samplesize=10
noise=0.05
q=10
g=2
z=5

X=pt.rand(samplesize,g,requires_grad=True)
Y=pt.rand(g,q,requires_grad=True)
Z=X@Y

test=a.VDAutoencoder(Z.shape, z).to(a.DEVICE)

print(X)
print(Z)

test.optimize(Z, 4000)