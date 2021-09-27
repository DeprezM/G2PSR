#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:48:32 2021

@author: mdeprez
"""

import sys
sys.path.append("..")

# from bayesNN import SNP_bnn
from bnn_GPSR import SNP_bnn
import time
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import r2_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import torch as pt
import pandas as pd
from gpu import DEVICE
import numpy as np

## Arguments order
# - directory 

# complete_filename = "/data/epione/user/mdeprez/benchmark_dataset/target_genes/" + str(sys.argv[1]) + "/"
complete_filename = "/user/mdeprez/home/Documents/BNN/benchmark_results/local_synthetic_dataset/200s_30g_15t_3tg_100tt_0.20n_0rep/"
var_str = "200s_30g_15t_3tg_100tt_0.20n_0rep"


X_csv=pd.read_csv(complete_filename+'gen_matrix.csv', sep=';',header=None)
X_Gsnp =pd.read_csv(complete_filename+'gen_snp.csv', sep=';',header=None)
W_csv = pd.read_csv(complete_filename+'w_snp_target.csv', sep=';',header=None)
Y = pd.read_csv(complete_filename+'target_noise.csv', sep=';',header=None).transpose()


# Change format 
X_group = X_Gsnp.to_numpy()
Y_tensor = pt.tensor(Y.to_numpy(), device=DEVICE, dtype=float)

X_tensor = []
for i in range(0,X_group.shape[1]):
    snpGene = X_group[:,i] == 1    
    Xi = pt.tensor(X_csv.loc[snpGene.tolist(),:].to_numpy().transpose(), device=DEVICE, dtype=float)
    X_tensor.append(Xi)


## Run BNN
start_time = time.time()
#data=SNP_bnn.gfptoCSV(args.nbsubject, args.genes, args.target, args.relevant, noise=args.noise)
bnn=SNP_bnn(X_tensor, Y_tensor)
# bnn.optimize(X_tensor, Y_tensor, epochmax=25000, step=1000, verbose=True)
result = bnn.optimize(X_tensor, Y_tensor, epochmax=50000, step=1000, verbose=True)
end_time = time.time()

# Check loss function
# loss_values = []
# for i in range(0, len(result["Losslist"])):
#     loss_values.append(result["Losslist"][i].item())
#np.savetxt(complete_filename+"/loss_values.csv", np.array(loss_values), delimiter=";")

# Ground truth relevant genes

info = var_str.split("_")
y = [1]*int(info[3][:-2]) + [0]*(int(info[1][:-1]) - int(info[3][:-2]))
testy = copy.deepcopy(bnn.probalpha().tolist())
testy = [1-i[0] for i in testy]

w_tmp = [np.array(i.weight.tolist())[0,:].tolist() for i in bnn.list_W_mu]
w_pred = []
for i in w_tmp:
    for el in i:
        w_pred.append(el)

data = {'w': W_csv.iloc[:,0].tolist(),
        'w_pred': w_pred}

df = pd.DataFrame(data = data)
R2 = r2_score(df.loc[:,'w'], df.loc[:,'w_pred'])


# Get perfomance metrics
compil_time = end_time - start_time
precision, recall, thresholds = precision_recall_curve(y,  testy)
avg_value = average_precision_score(y, testy)
auc_value = auc(recall, precision)
fscore = (2 * precision * recall) / (precision + recall)

ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))


prediction = [1 if el >= thresholds[ix] else 0 for el in testy]


confusion = confusion_matrix(y, prediction) 
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

print(accuracy_score(y , prediction)*100)







# values_name = ["results", "compilation_time", "precision", "recall", "thresholds", 
#                "Auc", "Avg", "Fscore", "R2",
#                "sample", "gene", "snps", "target", "target_gene",
#                "pct_target", "noise", "replicate", "file"]
# print("\t".join(values_name))
# for i in range(0, len(precision)-1):
#     list_of_features = ["result", round(compil_time), round(precision[i],2), round(recall[i],2), round(thresholds[i],2), 
#                         auc_value, avg_value, round(fscore[i],2), R2,
#                         float(info[0][:-1]), float(info[1][:-1]), X_group.shape[0], float(info[2][:-1]), float(info[3][:-2]),
#                         float(info[5][:-1]), float(info[4][:-2]), float(info[6][:-3]), var_str]
#     print("\t".join(map(str, list_of_features)))
