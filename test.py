# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:54 2020

@author: morei
"""


import autoencoder as a
import torch as pt
import numpy as np

#parameters
samplesize=15
q=10
g=3
z=5

#creating dummy data
X=pt.randn(samplesize,g,requires_grad=True)
temp=np.random.rand(g,q)
temp[2]=[0] * q
temp=temp.astype(float)
Y=pt.tensor(temp, requires_grad=True, dtype=pt.float)
Z=X@Y

# test=a.VDAutoencoder(Z.shape, z).to(a.DEVICE)

# print(X)
# print(Z)

# test.optimize(Z, 4000)

test=a.VDonWeightAE(Z).to(a.DEVICE)

test.optimize(Z, 4000)