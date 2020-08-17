# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:54 2020

@author: morei
"""


import autoencoder as a
import torch as pt
from autoencoder import SNPAutoencoder as snp

#for boxplot
import matplotlib.pyplot as plt

#%%
pt.cuda.empty_cache()
test=snp.genfullprofile(500, 20, 1, 1, noise=0.80)
X=test["X"]
Y=test["Y"]
pt.cuda.empty_cache()
test2=snp(X, Y).to(a.DEVICE)
test2.optimize(X, Y, 200)

#%%    
pt.cuda.empty_cache()
AutoEncoder=snp.CSVtoAutoEncoder()
AutoEncoder.optimize(epochmax=150000, step=100)

#%%
snplist=["rs7412","rs429358"]
featlist=["WholeBrain.bl","Ventricles.bl","Hippocampus.bl","MidTemp.bl","Entorhinal.bl",
          "CDRSB.bl", "ADAS11.bl", "MMSE.bl", "RAVLT.immediate.bl", "RAVLT.learning.bl", "RAVLT.forgetting.bl", "FAQ.bl"]
snp.boxplot(snplist, featlist)