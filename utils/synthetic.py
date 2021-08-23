#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:21:53 2021

@author: mdeprez
"""
import torch as pt
import numpy as np
import pandas as pd
from gpu import DEVICE # Check if GPU core available
import os
import copy


samplesize = 200
nb_gene = 20
nb_trait = 1
pct_trait = [100]#[10, 25, 50, 100]
nb_W2dim = 4#[1, 3, 9, 15, 30, 45, 60]
noise = [0.01, 0.4, 1]#, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]
nb_replicate = 1
filename = "/home/mdeprez/Documents/Data_ADNI/Simulation_results/Benchmark_datasets/Test/"#/data/epione/user/mdeprez/benchmark_dataset/noise_200G_mean_sd/"

nbSNPperGene = pd.read_csv('/home/mdeprez/Documents/Data_ADNI/pathways-bnn/nbSNPperGene.csv', names=("Gene", "Nb_snps"))
# nbSNPperGene = pd.read_csv('/data/epione/user/mdeprez/benchmark_dataset/nbSNPperGene.csv', names=("Gene", "Nb_snps"))

for r in range(0, nb_replicate):
    
    X=[]
    X_csv=[]
    X_Gsnp=[]
    nb_SNP = np.random.choice(nbSNPperGene.loc[:,"Nb_snps"].to_numpy(), size = nb_gene, replace=False)
    for i in range(0,nb_gene):
        # nb_SNP = np.random.choice(nbSNPperGene.loc[:,"Nb_snps"].to_numpy(), size = nb_gene, replace=False)
        # nb_SNP=np.round(abs(np.random.normal(cls.strand_mean,cls.strand_sd)))
        # while nb_SNP < 3:
        #     nb_SNP = np.round(abs(np.random.normal(cls.strand_mean,cls.strand_sd)))
        
        SNP=np.floor(abs(np.random.normal(scale = 1.5, size = (samplesize, nb_SNP[i]))))
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
        Wi=abs(np.random.randn(X[i].shape[1],nb_trait) )#* 2)
        if X[i].shape[1] > 3 :
            nb_snp_use = np.random.randint(3, X[i].shape[1], 1)
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
    Y = Y + pt.tensor([[1] * Y.shape[1]] * Y.shape[0], device=DEVICE)
    
    # ZZ = []
    # ww_csv = []
    for j in range(0, len(pct_trait)):
        ww_csv = []
        ZZ = []
        for g in range(0, nb_gene):
            trait_ind = int(np.floor(nb_trait*(pct_trait[j]/100)))
            ww = copy.deepcopy(W[g])
            ww[:, trait_ind:] = 0
            ww_csv.append(ww)
            ZZ.append(X[g] @ ww) 
        
        YY = sum(ZZ)
        # adapt the amplitude between noise and unknown ...
        mean_shift = np.mean(Y.numpy())
        std_shift = np.abs(np.random.normal(np.std(Y[:,0].numpy()), 1, Y[:, trait_ind:].shape[1]))
        YY[:, trait_ind:] = YY[:, trait_ind:] + np.random.normal(loc = [[mean_shift]*len(std_shift)] * samplesize, 
                                                                 scale = [std_shift] * samplesize)
        YY = YY + pt.tensor([[1] * YY.shape[1]] * YY.shape[0], device=DEVICE)
    
        for i in range(0, len(noise)):
            # noise_std = YY.numpy().std(0) * noise[i]
            # noise_mean = YY.numpy().mean(0) * noise[i]
            # noise_t=pt.normal(mean=pt.tensor([noise_mean * 1] * samplesize, dtype=pt.float, device=DEVICE), 
            #             std=pt.tensor([noise_std * 1] * samplesize, dtype=pt.float, device=DEVICE))
            
            # noise_mean = YY.numpy().mean(0) * noise[i]
            # noise_t=pt.normal(mean=pt.tensor([noise_mean * 1] * samplesize, dtype=pt.float, device=DEVICE), 
            #             std=pt.tensor([[1] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE))
            
            noise_std = YY.numpy().std(0) * noise[i]
            noise_t=pt.normal(mean=pt.tensor([[0] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE), 
                        std=pt.tensor([noise_std * 1] * samplesize, dtype=pt.float, device=DEVICE))
            
            Y_noise = copy.deepcopy(YY)
            Y_noise = Y_noise + noise_t
            
            
            complete_filename=filename+"%is_%ig_%it_%itg_%itt_%.2fn_%irep"%(samplesize, nb_gene, nb_trait, nb_W2dim, pct_trait[j], noise[i], r)
            if not os.path.exists(complete_filename):
                os.mkdir(complete_filename)
                np.savetxt(complete_filename+"/gen_matrix.csv", np.vstack(X_csv), delimiter=";")
                np.savetxt(complete_filename+"/gen_snp.csv", np.vstack(X_Gsnp), delimiter=";")
                np.savetxt(complete_filename+"/w_snp_target.csv", np.vstack(W_csv), delimiter=";")
                np.savetxt(complete_filename+"/target.csv", Y.numpy().transpose(), delimiter=";")
                np.savetxt(complete_filename+"/target_noise.csv", Y_noise.numpy().transpose(), delimiter=";")
    
