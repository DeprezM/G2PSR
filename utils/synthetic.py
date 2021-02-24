#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:21:53 2021

@author: mdeprez
"""
import torch as pt
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from gpu import DEVICE # Check if GPU core available
import os


samplesize = 400
nb_gene = 200
nb_trait = 15
nb_W2dim = 10
noise = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]
filename = "/home/mdeprez/Documents/Data_ADNI/Simulation_results/Benchmark_datasets/noise_same_3/"
 # def gfptoCSV(cls, samplesize, nb_gene, nb_trait, nb_W2dim, filename = "", noise=float(0.05), save_data = False):
nbSNPperGene = pd.read_csv('/home/mdeprez/Documents/Data_ADNI/pathways-bnn/nbSNPperGene.csv', names=("Gene", "Nb_snps"))

X=[]
X_csv=[]
X_Gsnp=[]
nb_SNP = np.random.choice(nbSNPperGene.loc[:,"Nb_snps"].to_numpy(), size = nb_gene, replace=False)
for i in range(0,nb_gene):
    # nb_SNP = np.random.choice(nbSNPperGene.loc[:,"Nb_snps"].to_numpy(), size = nb_gene, replace=False)
    # nb_SNP=np.round(abs(np.random.normal(cls.strand_mean,cls.strand_sd)))
    # while nb_SNP < 3:
    #     nb_SNP = np.round(abs(np.random.normal(cls.strand_mean,cls.strand_sd)))
    
    SNP=np.floor(abs(np.random.randn(samplesize, nb_SNP[i])))
    for sample in range(0,SNP.shape[0]):
        for snp in range(0,SNP.shape[1]):
            if SNP[sample][snp]>2: 
                SNP[sample][snp]=2
    X_csv.append(SNP.transpose())
    gn_snp = np.zeros((SNP.shape[1], nb_gene))
    gn_snp[:,i] = 1
    
    SNP = pt.tensor(SNP, device=DEVICE, dtype=float)
    X.append(SNP)
    X_Gsnp.append(gn_snp)
W=[]
W_csv=[]
Z=[]
for i in range(nb_W2dim):
    Wi=abs(np.random.randn(X[i].shape[1],nb_trait) * 5)
    if X[i].shape[1] > 5 :
        nb_snp_use = np.random.randint(5, X[i].shape[1], 1)
        Wi[-(X[i].shape[1]-nb_snp_use[0]):,:]=0
    W_csv.append(Wi)
    Wi=pt.tensor(Wi, device=DEVICE, dtype=float)
    W.append(Wi)
    Z.append(X[i] @ Wi)
for i in range(nb_W2dim, nb_gene):
    Wi=[[0] * nb_trait] * X[i].shape[1]
    W_csv.append(Wi)
    Wi=pt.tensor(Wi, device=DEVICE, dtype=float)
    W.append(Wi)
    Z.append(X[i] @ Wi)

Y=sum(Z)
Y=Y + pt.tensor([[1] * Y.shape[1]] * Y.shape[0], device=DEVICE)

for i in range(0, len(noise)):
    noise_std = np.std(Y.numpy()) * noise[i]
    noise_t=pt.normal(mean=pt.tensor([[0] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE), 
                std=pt.tensor([[noise_std * 1] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE))
    Y_noise = Y + noise_t
        
    complete_filename=filename+"%is_%ig_%it_%itg_%.2fn"%(samplesize, nb_gene, nb_trait, nb_W2dim, noise[i])
    if not os.path.exists(complete_filename):
        os.mkdir(complete_filename)
        np.savetxt(complete_filename+"/gen_matrix.csv", np.vstack(X_csv), delimiter=";")
        np.savetxt(complete_filename+"/gen_snp.csv", np.vstack(X_Gsnp), delimiter=";")
        np.savetxt(complete_filename+"/w_snp_target.csv", np.vstack(W_csv), delimiter=";")
        np.savetxt(complete_filename+"/target.csv", Y.numpy().transpose(), delimiter=";")
        np.savetxt(complete_filename+"/target_noise.csv", Y_noise.numpy().transpose(), delimiter=";")

