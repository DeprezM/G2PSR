#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:49:15 2021

@author: mdeprez
"""

import sys
sys.path.append("..")

from bayesNN import SNP_bnn
import time
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import torch as pt
import pandas as pd
from gpu import DEVICE


### % Argument parser

## Arguments order
# - directory 

# complete_filename = "/home/mdeprez/benchmark/test_data/noise/" + str(sys.argv[1]) + "/"
complete_filename = "/home/mdeprez/Documents/Data_ADNI/Simulation_results/Benchmark_datasets/noise2/400s_200g_15t_2tg_0.01n/"

X_csv=pd.read_csv(complete_filename+'gen_matrix.csv', sep=';',header=None)
X_Gsnp =pd.read_csv(complete_filename+'gen_snp.csv', sep=';',header=None)
W_csv = pd.read_csv(complete_filename+'w_snp_target.csv', sep=';',header=None)
Y = pd.read_csv(complete_filename+'target.csv', sep=';',header=None).transpose()

# Change format 
X_group = X_Gsnp.to_numpy()

X_tensor = []
for i in range(0,X_group.shape[1]):
    snpGene = X_group[:,i] == 1    
    Xi = pt.tensor(X_csv.loc[snpGene.tolist(),:].to_numpy().transpose(), device=DEVICE, dtype=float)
    X_tensor.append(Xi)

Y_tensor = []
for i in range(0, Y.shape[0]):
    Y_tensor.append(pt.tensor(Y.iloc[i,:].to_numpy(), device = DEVICE, dtype=float))


Y_tensor = pt.tensor(Y.to_numpy(), device = DEVICE, dtype=float)
## Run BNN
start_time = time.time()
# data=SNP_bnn.gfptoCSV(400, 200, 15, 10, noise=0.1)
# len(data["X"])
# data["X"][0].shape
X_tensor[0].shape

# len(data["Y"])
# data["Y"][0].shape
Y_tensor[0].shape
# bnn=SNP_bnn(data["X"], data["Y"])
bnn=SNP_bnn(X_tensor, Y_tensor)
bnn.optimize(X_tensor, Y_tensor, epochmax=100000, step=1000)
end_time = time.time()

# Ground truth relevant genes
var_str = str(sys.argv[1]) 
info = var_str.split("_")
y = [0]*int(info[3][:-2]) + [1]*(int(info[1][:-1]) - int(info[3][:-2]))

# Get perfomance metrics
compil_time = end_time - start_time
precision, recall, thresholds = precision_recall_curve(y,  bnn.probalpha().tolist())
auc = auc(recall, precision)

values_name = ["compilation_time", "precision", "recall", "thresholds", "Auc", "Noise"]
print("\t".join(values_name))
for i in range(0, len(precision)-1):
    print("\t".join(map(str, [round(compil_time,2), round(precision[i],2), round(recall[i],2), thresholds[i], auc, float(info[4][:-1])])))
