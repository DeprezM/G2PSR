#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:47:12 2021

@author: mdeprez
"""


import sys
sys.path.append("..")

from bnn_GPSR import SNP_bnn
import time
import torch as pt
import pandas as pd
from gpu import DEVICE
import numpy as np

## Arguments order
# - directory 

complete_filename = "/home/mdeprez/test_adni/3_pathways/data/" 
# complete_filename = "/user/mdeprez/home/Documents/Data_ADNI/adni_datasets/3_pathways/" 


##### Load files -------------------------------------------------------------
X_csv=pd.read_csv(complete_filename+'gen_matrix.csv', sep=';',header=None)
X_Gsnp =pd.read_csv(complete_filename+'gen_snp.csv', sep=';',header=None)
Y = pd.read_csv(complete_filename+'target.csv', sep=';',header=None)
X_names = pd.read_csv(complete_filename+'gen_snp_names.csv', sep=';')
Y_names = ["Hippocampus.bl","Entorhinal.bl", "CDRSB.bl","ADAS11.bl","MMSE.bl","RAVLT.immediate.bl","RAVLT.forgetting.bl","FAQ.bl"]

##### Adapt format -----------------------------------------------------------
X_group = X_Gsnp.to_numpy()
X_group = X_group[[i == 1 for i in np.sum(X_group, axis = 1)],:]
Y_tensor = pt.tensor(Y.to_numpy(), device=DEVICE, dtype=float)

X_tensor = []
for i in range(0,X_group.shape[1]):
    snpGene = X_group[:,i] == 1    
    Xi = pt.tensor(X_csv.loc[snpGene.tolist(),:].to_numpy().transpose(), device=DEVICE, dtype=float)
    X_tensor.append(Xi)


##### Run BNN ----------------------------------------------------------------
start_time = time.time()
bnn=SNP_bnn(X_tensor, Y_tensor)
result = bnn.optimize(X_tensor, Y_tensor, epochmax=50000, step=1000)
end_time = time.time()

##### Optimization and final values
# Check loss function
for i in range(0, len(result["Losslist"])):
    list_of_features = ["Loss", result["Losslist"][i].item(), 
                         X_group.shape[0], X_group.shape[1], Y_tensor.shape[0], Y_tensor.shape[1]]
    print("\t".join(map(str, list_of_features)))

# Check prob Alpha
for i in range(0, result["plist"].shape[0]):
    list_of_features = ["ProbAlpha"]
    for j in range(0, result["plist"].shape[1]):
        list_of_features.append(result["plist"][i, j])
    list_of_features += [X_names.columns[i] ,X_group.shape[0], X_group.shape[1], Y_tensor.shape[0], Y_tensor.shape[1]]
    print("\t".join(map(str, list_of_features)))

  
# Check mulist
for i in range(0, result["mulist"].shape[0]):
    list_of_features = ["mu_param"]
    for j in range(0, result["mulist"].shape[1]):
        list_of_features.append(result["mulist"][i, j])
    list_of_features += [X_names.columns[i], X_group.shape[0], X_group.shape[1], Y_tensor.shape[0], Y_tensor.shape[1]]
    print("\t".join(map(str, list_of_features)))

# Check mulist per snp
for i in range(0, len(result["musnplist"][0])):
    list_of_features = ["mu_snp_param"]
    for j in range(0, len(result["musnplist"])):
        list_of_features.append(result["musnplist"][j][i])
    list_of_features += [X_names.index[i], X_group.shape[0], X_group.shape[1], Y_tensor.shape[0], Y_tensor.shape[1]]
    print("\t".join(map(str, list_of_features)))
    
    # Check Y noise
for i in range(0, len(result["ynoise"][0])):
    list_of_features = ["ynoise_param"]
    for j in range(0, len(result["ynoise"])):
        list_of_features.append(result["ynoise"][j][i])
    list_of_features += [ Y_names[i], X_group.shape[1], Y_tensor.shape[0], Y_tensor.shape[1]]
    print("\t".join(map(str, list_of_features)))
    
# Check Y noise
for i in range(0, len(result["ybias"][0])):
    list_of_features = ["ybias_param"]
    for j in range(0, len(result["ybias"])):
        list_of_features.append(result["ybias"][j][i])
    list_of_features += [Y_names[i], X_group.shape[1], Y_tensor.shape[0], Y_tensor.shape[1]]
    print("\t".join(map(str, list_of_features)))

##### Results ----------------------------------------------------------------
# Computation time
compil_time = end_time - start_time

print("time %f"%compil_time)