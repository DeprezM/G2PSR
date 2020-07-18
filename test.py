# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:54 2020

@author: morei
"""


import autoencoder as a
import torch as pt
import numpy as np
from autoencoder import SNPAutoencoder as snp

#for boxplot
import matplotlib.pyplot as plt

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
    
#def test2():
test=snp.genfullprofile(500, 4, 11, 1, noise=0.05)
X=test["X"]
Y=test["Y"]

test2=snp(X, Y).to(a.DEVICE)
#stat=test["W2"] * test["Z"].transpose(0,1)
test2.optimize(X, Y, 10000)
print(test["W2"])
#print(stat.mean(1))
    
# pt.cuda.empty_cache()
# AutoEncoder=snp.CSVtoAutoEncoder()
# AutoEncoder.optimize(epochmax=60000, step=100)

# data=snp.loadData()["data"]
# snplist=["rs72654468","rs1295687"]
# featlist=["WholeBrain.bl","Ventricles.bl","Hippocampus.bl","MidTemp.bl","Entorhinal.bl",
#           "CDRSB.bl", "ADAS11.bl", "MMSE.bl", "RAVLT.immediate.bl", "RAVLT.learning.bl", "RAVLT.forgetting.bl", "FAQ.bl"]
# for i in range(len(snplist)):
#     rssnp=snplist[i]
#     for i2 in range(len(featlist)):
#         vol=featlist[i2]
#         temp=data[[rssnp,vol]]
#         fig,ax=plt.subplots()
#         ax.set_title(rssnp+" "+vol)
#         ax.boxplot([temp[temp[rssnp] == 0][vol], 
#                     temp[temp[rssnp] == 1][vol], 
#                     temp[temp[rssnp] == 2][vol]],
#                     labels=[0,1,2])
# plt.close()