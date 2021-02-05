#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:05:22 2021

@author: mdeprez
"""

import sys
sys.path.append("..")

import bayesNN as a
import torch as pt
from bayesNN import SNP_bnn as snp

#for boxplot
import matplotlib.pyplot as plt
import pandas as pd
import itertools

#%% Test on synthetic data

pt.cuda.empty_cache()
test=snp.gfptoCSV(200, 30, 20, 5, noise=0.1, 
                  filename = "/home/mdeprez/Documents/Data_ADNI/Simulation_results/Benchmark_datasets/") # 500 samples, 160 genes
X=test["X"]
Y=test["Y"]

#Y=test["W"]

pt.cuda.empty_cache()
test2=snp(X, Y).to(a.DEVICE)
result = test2.optimize(X, Y, epochmax=30000, step=100)


y = [0]*5 + [1]*25

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y, test2.probalpha().tolist())

# calculate AUC
auc = roc_auc_score(y, test2.probalpha().tolist())
print('AUC: %.3f' % auc)

test2.probalpha().tolist()


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

precision, recall, thresholds = precision_recall_curve(y,  test2.probalpha().tolist())
# calculate precision-recall AUC
auc = auc(recall, precision)

# ## Check if true genes were found.
sortedAlpha, indices = pt.sort(test2.probalpha(), 0)

pmin = sortedAlpha[4]
# pmin = (test2.probalpha().mean() - test2.probalpha().min()) * 0.05
# pmin = test2.probalpha()[0:10].max()
# pmin = test2.probalpha().min()*5

TP=0 #true positives
TN=0 #true negatives
FP=0 #false positives
FN=0 #false negatives
  #for the distinction between a SNP assumed relevant and one not
for i2 in range(3):
    if test2.probalpha()[i2]<=pmin: TP+=1
    else: FN+=1
for i2 in range(3,20):
    if test2.probalpha()[i2]<=pmin: FP+=1
    else: TN+=1





# pt.cuda.empty_cache()
# test=snp.genfullprofile(500, 500, 11, 1, noise=0.05)
# X=test["X"]
# Y=test["Y"]
# pt.cuda.empty_cache()

# pt.cuda.empty_cache()
# AutoEncoder=snp.CSVtoAutoEncoder()


# test=snp(AutoEncoder.X, AutoEncoder.Y).to(a.DEVICE)
# result = test.optimize(AutoEncoder.X, AutoEncoder.Y, 60000)

# # pt.save(test2,"/home/mdeprez/Documents/Data_ADNI/RealData_results/500G_test.pt" )
# #%2%    

# geneName = []
# for i in range(0,len(AutoEncoder.dfX)):
#     geneName.append(AutoEncoder.dfX[i][0])

# with open("/home/mdeprez/Documents/Data_ADNI/RealData_results/500G_snpValues.txt", 'w') as f:
#     for item in snpValues:
#         f.write("%s\n" % item)


# snpNames = []
# snpValues = []
# for i in range(0, len(AutoEncoder.dfX)):
#     snpNames.append(AutoEncoder.dfX[i][1].columns.tolist())
#     snpValues = snpValues + test2.list_W_mu[i].weight.data.flatten().tolist()

# snpNames = list(itertools.chain.from_iterable(snpNames))


# #%%
# plot_values = AutoEncoder.optimize(epochmax=3, step=1)

# AutoEncoder=snp.CSVtoAutoEncoder()
# AutoEncoder.optimize(epochmax=150000, step=100)

