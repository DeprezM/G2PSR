#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:49:15 2021

@author: mdeprez
"""

import sys
sys.path.append("..")

from bnn_GPSR import SNP_bnn
import time
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import torch as pt
import pandas as pd
from gpu import DEVICE
import numpy as np

## Arguments order
# - directory 

complete_filename = "/data/epione/user/mdeprez/benchmark_dataset/samples/" + str(sys.argv[1]) + "/"

##### Load files -------------------------------------------------------------
X_csv=pd.read_csv(complete_filename+'gen_matrix.csv', sep=';',header=None)
X_Gsnp =pd.read_csv(complete_filename+'gen_snp.csv', sep=';',header=None)
W_csv = pd.read_csv(complete_filename+'w_snp_target.csv', sep=';',header=None)
Y = pd.read_csv(complete_filename+'target_noise.csv', sep=';',header=None).transpose()

var_str = str(sys.argv[1]) 
info = var_str.split("_")
y = [1]*int(info[3][:-2]) + [0]*(int(info[1][:-1]) - int(info[3][:-2]))

##### Adapt format -----------------------------------------------------------
X_group = X_Gsnp.iloc[:,:int(info[1][:-1])].to_numpy()
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

##### Optimization
# Check loss function
for i in range(0, len(result["Losslist"])):
    list_of_features = ["Loss", result["Losslist"][i].item(), 
                        float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
                        float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
    print("\t".join(map(str, list_of_features)))

# Check prob Alpha
for i in range(0, result["plist"].shape[0]):
    list_of_features = ["ProbAlpha"]
    for j in range(0, result["plist"].shape[1]):
        list_of_features.append(result["plist"][i, j])
    list_of_features += ["G_"+str(i+1) ,float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
                         float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
    print("\t".join(map(str, list_of_features)))

# Check probAlpha _ performance per epoch
for i in range(0, result["plist"].shape[1]):
    list_of_features = ["Palpha_auc", i*1000]
    testy = copy.deepcopy(result["plist"][:, i].tolist())
    testy = [1-j for j in testy]
    precision, recall, thresholds = precision_recall_curve(y,  testy)
    avg_value = average_precision_score(y, testy)
    auc_value = auc(recall, precision)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    prediction = [1 if el >= thresholds[ix] else 0 for el in testy]
    confusion = confusion_matrix(y, prediction) 
    FN = confusion[1][0]
    TN = confusion[0][0]
    TP = confusion[1][1]
    FP = confusion[0][1]
    accuracy = accuracy_score(y, prediction)*100

    list_of_features += [auc_value, avg_value, round(fscore[ix],2),
                         TP, TN, FP, FN, accuracy,
                         float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
                         float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
    print("\t".join(map(str, list_of_features)))    

# Check mulist
for i in range(0, result["mulist"].shape[0]):
    list_of_features = ["mu_param"]
    for j in range(0, result["mulist"].shape[1]):
        list_of_features.append(result["mulist"][i, j])
    list_of_features += ["G_"+str(i+1) ,float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
                         float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
    print("\t".join(map(str, list_of_features)))

# Check Y noise
for i in range(0, len(result["ynoise"][0])):
    list_of_features = ["ynoise_param"]
    for j in range(0, len(result["ynoise"])):
        list_of_features.append(result["ynoise"][j][i])
    list_of_features += ["P_"+str(i+1) ,float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
                         float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
    print("\t".join(map(str, list_of_features)))
    
# Check Y noise
for i in range(0, len(result["ybias"][0])):
    list_of_features = ["ybias_param"]
    for j in range(0, len(result["ybias"])):
        list_of_features.append(result["ybias"][j][i])
    list_of_features += ["P_"+str(i+1) ,float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
                         float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
    print("\t".join(map(str, list_of_features)))
    
    
# Check mulist per snp
for i in range(0, len(result["musnplist"][0])):
    list_of_features = ["mu_snp_param"]
    for j in range(0, len(result["musnplist"])):
        list_of_features.append(result["musnplist"][j][i])
    list_of_features += ["SNP_"+str(i+1) ,float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
                         float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
    print("\t".join(map(str, list_of_features)))

##### Results ----------------------------------------------------------------
# Computation time
compil_time = end_time - start_time

# Results
testy = copy.deepcopy(bnn.probalpha().tolist())
testy = [1-i[0] for i in testy]

# Performance metrics
precision, recall, thresholds = precision_recall_curve(y,  testy)
avg_value = average_precision_score(y, testy)
auc_value = auc(recall, precision)
fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)

prediction = [1 if el >= thresholds[ix] else 0 for el in testy]

confusion = confusion_matrix(y, prediction) 
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

accuracy = accuracy_score(y, prediction)*100

# SNP level
# Print results per SNPs and their genes
w_tmp = [np.array(i.weight.tolist())[0,:].tolist() for i in bnn.list_W_mu]
w_real = W_csv.mean(axis=1)
snp_binary = [1 if el != 0 else 0 for el in w_real]
w_pred = []
w_pred_all = []
gene_idx = []
gene_alpha = []
gene_mu = []
for idx, val in enumerate(w_tmp):
    w_pred_all += val
    gene_idx += [idx]*len(val)
    gene_alpha += [result["plist"][idx,-1]]*len(val)
    gene_mu += [result["mulist"][idx,-1]]*len(val)
    if prediction[idx] == 1:
        w_pred += val
    else:
        w_pred += [0]*len(val)

for i in range(0, len(w_pred)):
    list_of_features = ["w_snps", w_real[i], w_pred_all[i], w_pred[i], gene_idx[i], y[idx], gene_alpha[idx], gene_mu[idx],
                        float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
                        float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
    print("\t".join(map(str, list_of_features)))


precision_snp, recall_snp, thresholds_snp = precision_recall_curve(snp_binary, w_pred_all)
avg_value_snp = average_precision_score(snp_binary, w_pred_all)
auc_value_snp = auc(recall_snp, precision_snp)
fscore_snp = (2 * precision_snp * recall_snp) / (precision_snp + recall_snp)
ix_snp = np.argmax(fscore_snp)

prediction_snp = [1 if el >= thresholds_snp[ix_snp] else 0 for el in w_pred_all]

confusion = confusion_matrix(snp_binary, prediction_snp) 
FN_snp = confusion[1][0]
TN_snp = confusion[0][0]
TP_snp = confusion[1][1]
FP_snp = confusion[0][1]

accuracy_snp = accuracy_score(snp_binary, prediction_snp)*100

## Print results 
values_name = ["results", "compilation_time", "precision", "recall", "threshold", 
               "Auc", "Avg", "Fscore",
               "TP", "TN", "FP", "FN", "accuracy",
               "precision_snp", "recall_snp", "threshold_snp", 
               "Auc_snp", "Avg_snp", "Fscore_snp",
               "TP_snp", "TN_snp", "FP_snp", "FN_snp", "accuracy_snp",
               "sample", "gene", "snps", "relevant_snps", "target", "target_gene",
               "pct_target", "noise", "replicate", "file"]
print("\t".join(values_name))

list_of_features = ["result", round(compil_time), round(precision[ix],2), round(recall[ix],2), round(thresholds[ix],2), 
                    auc_value, avg_value, round(fscore[ix],2),
                    TP, TN, FP, FN, accuracy,
                    round(precision_snp[ix_snp],2), round(recall_snp[ix_snp],2), round(thresholds_snp[ix_snp],2), 
                    auc_value_snp, avg_value_snp, round(fscore_snp[ix_snp],2),
                    TP_snp, TN_snp, FP_snp, FN_snp, accuracy_snp,
                    float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], sum(snp_binary), float(info[2][:-1]), float(info[3][:-2]),
                    float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
print("\t".join(map(str, list_of_features)))
