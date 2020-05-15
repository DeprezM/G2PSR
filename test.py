# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:54 2020

@author: morei
"""


import autoencoder as a
import torch as pt
import numpy as np

#parameters
samplesize=500
q=20
g=10
z=5

#creating dummy data
X=pt.randn(samplesize,g)
temp=np.random.rand(g,q)
temp[2]=[0] * q
temp=temp.astype(float)
Y=pt.tensor(temp, dtype=pt.float)
Z=X@Y

# test=a.VDAutoencoder(Z.shape, z).to(a.DEVICE)

# print(X)
# print(Z)

# test.optimize(Z, 4000)

test=a.VDonWeightAE(X,Z).to(a.DEVICE)

test.optimize(X, Z, 4000)