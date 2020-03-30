# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:54 2020

@author: morei
"""


import autoencoder as a
import torch as pt

samplesize=30
noise=0.05
q=3
d=2

X=pt.rand(q,samplesize,requires_grad=True)

test=a.autoencoder(X.shape, a.q).to(a.DEVICE)

print(X)

test.optimize(X, 1000)