#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:07:57 2021

@author: mdeprez
"""

import torch as pt
import numpy as np
import pandas as pd
from gpu import DEVICE # Check if GPU core available
import os
import copy

##### Parameter values -------------------------------------------------------
samplesize_list = [500] # Number of samples
nb_gene_list = [200] # Number of genes (total)
nb_trait_list = [1, 2, 5, 10, 15, 20, 30] # Number of phenotypic traits
nb_true_trait = [20, 50, 100] # Percentage of relevant phenotypic traits
nb_true_gene_list = [5] # Number of relevant genes
noise = [0.2] # Noise level (percentage of phenotypic sd)
nb_replicate = 10 # Number of different dataset

# nbSNPperGene = pd.read_csv('/home/mdeprez/Documents/Data_ADNI/pathways-bnn/nbSNPperGene.csv', names=("Gene", "Nb_snps"))
nbSNPperGene = pd.read_csv('/data/epione/user/mdeprez/benchmark_dataset/nbSNPperGene.csv', names=("Gene", "Nb_snps"))


##### Output directory
filename = "/data/epione/user/mdeprez/benchmark_dataset/pheno_target/"
# filename = "/user/mdeprez/home/Documents/Data_ADNI/Simulation_results/Benchmark_datasets/"


for r in range(0, nb_replicate):
    print("Generating data... %i"%r)
    
##### Genotype matrix --------------------------------------------------------
    X=[None]*np.max(nb_gene_list)
    X_csv=[None]*np.max(nb_gene_list)
    X_Gsnp=[None]*np.max(nb_gene_list)
    nb_SNP = np.random.choice(nbSNPperGene.loc[:,"Nb_snps"].to_numpy(), size = np.max(nb_gene_list), replace=False)
    
    for i in range(0, np.max(nb_gene_list)):
        if i % 500 == 0:
            print(i)
        # sum_snp_value = np.random.multinomial(100000000, [0.56, 0.29, 0.15]) # Proportion of each variant status (ADNI reference {0,1,2})
        # example_snp_value = np.array([0]*sum_snp_value[0] + [1]*sum_snp_value[1] + [2]*sum_snp_value[2])
        # np.random.shuffle(example_snp_value)
        SNP = np.random.choice([0,1,2], size = (np.max(samplesize_list), nb_SNP[i]), replace=True, p = [0.56, 0.29, 0.15])
        X_csv[i] = SNP.transpose()
        gn_snp = np.zeros((SNP.shape[1], np.max(nb_gene_list)))
        gn_snp[:,i] = 1
        
        SNP = pt.tensor(SNP, device=DEVICE, dtype=float)
        X[i] = SNP
        X_Gsnp[i] = gn_snp

##### Linear transformation matrix -------------------------------------------
    print("Other parameter included ...")
    for pheno in range(len(nb_trait_list)):
        nb_trait = nb_trait_list[pheno]
        for tg in range(len(nb_true_gene_list)):
            nb_true_gene = nb_true_gene_list[tg]
            W=[]
            W_csv=[]
            Z=[]
            for i in range(nb_true_gene):
                Wi=abs(np.random.randn(X[i].shape[1], nb_trait) )#* 2)
                if X[i].shape[1] >= 5 : # number of relevant SNPs
                    if np.random.randint(1, 10, 1) < 3 :
                        nb_snp_use = np.random.randint(1, X[i].shape[1], 1) # np.random.randint(5, X[i].shape[1], 1)
                    else : 
                        nb_snp_use = np.random.randint(1, 4, 1) # np.random.randint(5, X[i].shape[1], 1)
                    Wi[-(X[i].shape[1]-nb_snp_use[0]):,:]=0
                elif X[i].shape[1] < 5 and X[i].shape[1] > 2:
                    nb_snp_use = np.random.randint(1, X[i].shape[1], 1) # if small number of SNPs in selected gene
                    Wi[-(X[i].shape[1]-nb_snp_use[0]):,:]=0
                W_csv.append(Wi)
                Wi=pt.tensor(Wi, device=DEVICE, dtype=float)
                W.append(Wi)
                Z.append(X[i] @ Wi)
            for i in range(nb_true_gene, np.max(nb_gene_list)):
                Wi=[[0] * nb_trait] * X[i].shape[1]
                W_csv.append(Wi)
                Wi=pt.tensor(Wi, device=DEVICE, dtype=float)
                W.append(Wi)
                Z.append(X[i] @ Wi)
    
##### Phenotypic matrix ------------------------------------------------------
            for nb_gene in nb_gene_list:
                Y=sum(Z[:nb_gene])
                Y = Y + pt.tensor([[1] * Y.shape[1]] * Y.shape[0], device=DEVICE)
            
                for smp in range(0, len(samplesize_list)):
                    samplesize = samplesize_list[smp]
                    XX_csv = copy.deepcopy(X_csv[:nb_gene])
                    for j in range(0, len(nb_true_trait)):
                        ww_csv = []
                        ZZ = []
                        for g in range(0, nb_gene):
                            trait_ind = int(np.floor(nb_trait*(nb_true_trait[j]/100)))
                            ww = copy.deepcopy(W[g])
                            ww[:, trait_ind:] = 0
                            # ww[:, (nb_true_trait[j]-1):] = 0
                            ww_csv.append(ww)
                            ZZ.append(X[g][:samplesize,:] @ ww)
                            XX_csv[g] = X_csv[g][:,:samplesize]
                        
                        YY = sum(ZZ)
                        # adapt the amplitude between noise and unknown ...
                        mean_shift = np.mean(YY[:, :trait_ind].numpy())
                        std_shift = abs(np.random.normal(np.std(YY[:,0].numpy()), 1, YY[:, trait_ind:].shape[1]))
                        YY[:, trait_ind:] = YY[:, trait_ind:] + np.random.normal(loc = [[mean_shift]*len(std_shift)] * samplesize, 
                                                                                  scale = [std_shift] * samplesize)
                        YY = YY + pt.tensor([[1] * YY.shape[1]] * YY.shape[0], device=DEVICE)
                        # mean_shift = np.mean(YY[:, :(nb_true_trait[j]-1)].numpy())
                        # std_shift = abs(np.random.normal(np.std(YY[:,0].numpy()), 1, YY[:, (nb_true_trait[j]-1):].shape[1]))
                        # YY[:, (nb_true_trait[j]-1):] = YY[:, (nb_true_trait[j]-1):] + np.random.normal(loc = [[mean_shift]*len(std_shift)] * samplesize, 
                        #                                                           scale = [std_shift] * samplesize)
                        # YY = YY + pt.tensor([[1] * YY.shape[1]] * YY.shape[0], device=DEVICE)
                    
##### Noise ------------------------------------------------------------------
                        for i in range(0, len(noise)):
                            noise_std = YY.numpy().std(0) * noise[i]
                            noise_t=pt.normal(mean=pt.tensor([[0] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE), 
                                        std=pt.tensor([noise_std * 1] * samplesize, dtype=pt.float, device=DEVICE))
                            
                            Y_noise = copy.deepcopy(YY)
                            Y_noise = Y_noise + noise_t
                            
                            complete_filename=filename+"%is_%ig_%it_%itg_%itt_%.2fn_%irep"%(samplesize, nb_gene, nb_trait, nb_true_gene, nb_true_trait[j], noise[i], r)
                            if not os.path.exists(complete_filename):
                                os.mkdir(complete_filename)
                                np.savetxt(complete_filename+"/gen_matrix.csv", np.vstack(XX_csv), delimiter=";")
                                np.savetxt(complete_filename+"/gen_snp.csv", np.vstack(X_Gsnp[:nb_gene]), delimiter=";")
                                np.savetxt(complete_filename+"/w_snp_target.csv", np.vstack(ww_csv), delimiter=";")
                                np.savetxt(complete_filename+"/target.csv", YY.numpy().transpose(), delimiter=";")
                                np.savetxt(complete_filename+"/target_noise.csv", Y_noise.numpy().transpose(), delimiter=";")
                
