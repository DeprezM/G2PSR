# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:54 2020

@author: morei
"""


import autoencoder as a
import torch as pt
import numpy as np
from autoencoder import SNPAutoencoder as snp

def test1():
    #parameters
    samplesize=500
    q=20
    g=10
    noise=5
    
    #creating dummy data
    X=pt.randn(samplesize,g)
    temp=np.random.rand(g,q)
    temp[2]=[0] * q
    temp[4]=[0] * q
    temp[7]=[0] * q
    temp=temp.astype(float)
    Y=pt.tensor(temp, dtype=pt.float)
    Ztrue=X@Y
    noise=pt.normal(mean=pt.Tensor([[0] * q] * samplesize), 
                    std=pt.Tensor([[noise] * q] * samplesize))
    Z=Ztrue+noise
    
    print(Ztrue)
    print(Z)
    print(pt.mul(Z-Ztrue, 1/Z))
    
    test=a.VDonWeightAE(X,Z).to(a.DEVICE)
    test.optimize(X, Z, 5000)
    
    return test
    
def test2():
    test=snp.genfullprofile(2000,10, 3, 7)
    X=test["X"]
    Y=test["Y"]
    
    test2=snp(X, Y).to(a.DEVICE)
    #stat=test["W2"] * test["Z"].transpose(0,1)
    test2.optimize(X, Y, 10000)
    print(test["W2"])
    #print(stat.mean(1))
    

AutoEncoder=snp.CSVtoAutoEncoder()
AutoEncoder.optimize(epochmax=1000000, step=100)