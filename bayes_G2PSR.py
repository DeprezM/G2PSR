#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:26:11 2021

@author: mdeprez
"""

## Bayesian Neural Network for Genome to Phenome Sparse Regression --> G2PSR

import torch as pt
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from plotnine import *
from gpu import DEVICE # Check if GPU core available
import os

class G2PSR(pt.nn.Module):
    _pmin= 0.05     # the value of p under which we keep the gene for analysis of its SNP
    _bias=1         # the average value of the generated physiological traits


#----------------------------------------------------------------------------------------------------------------------------------------------    
    # Initialisation function
    ## Input parameters :
    ##      self - object of class G2PSR
    ##      input_list - tensor containing genetic data         
    ##      output_shape - tensor containing physiologic data
    ##      Default initialisation parameters used to define the neural network
    ##      The use of fixed values avoid the unbalance between features in later optimization
    ##      as opposed to a random initialization of these parameters
    def __init__(self, input_list, output_shape, mu0=1, alpha0=0, logvar0=0.5, bias0=0, noise0=0.5):
        super().__init__()
        
        # Obtaining the dimension
        listdim=[] # list of the SNPs dimensions associated with each gene 
        self.X=None
        self.Y=None
        self.dfX=None
        self.dfY=None
        
        if type(input_list) is list:
            for i in range(0,len(input_list)):
                if type(input_list[i]) is pt.Tensor:
                    if (self.X is None): self.X=[]
                    input_shape=input_list[i].shape
                    if len(input_shape) == 1 :
                        listdim.append(1)
                    else:
                        listdim.append(input_shape[1])
                    self.X.append(input_list[i])
                else: return("error")
        else: return("error")
        
        if type(output_shape) is pt.Tensor:
            self.Y=output_shape
            output_shape=output_shape.shape
        if type(output_shape) is pt.Size or type(output_shape) is list:
            if (len(output_shape)==1):
                outputdim=1
            else: outputdim=output_shape[1]
        elif type(output_shape) is int:
            outputdim=output_shape
        else: return("error")
        
        
        # Creating the variational parameters using the distribution of a linear transformation in the codding hidden layer
        list_W_mu=[]
        list_W_logvar=[]
        
        for i in range(0, len(listdim)):
            mu = pt.nn.Linear(listdim[i], 1, bias=False).to(DEVICE)
            mu.weight = pt.nn.Parameter(pt.tensor([[mu0] * listdim[i]], dtype=pt.float, device=DEVICE))
            
            logvar = pt.nn.Linear(listdim[i],1, bias=False).to(DEVICE)
            logvar.weight = pt.nn.Parameter(pt.tensor([[logvar0] * listdim[i]], dtype=pt.float, device=DEVICE))
            
            list_W_mu.append(mu)
            list_W_logvar.append(logvar)
            
        self.list_W_mu=list_W_mu
        self.list_W_logvar=list_W_logvar
        
        # Creating sigma parameter for decoding layer estimate
        self.noise = pt.nn.Parameter(pt.Tensor([[noise0] * output_shape[1]]).to(DEVICE), requires_grad=True) # dimension of the output
        
        # Creating the parameters in the decoding part (dropout with log(alpha), mu' and the bias associated with the physiological traits)
        self.alpha=pt.nn.Parameter(pt.Tensor([[alpha0]] * len(listdim)).to(DEVICE), requires_grad=True) # alpha is a log
        self.mu=pt.nn.Parameter(pt.Tensor([[mu0] * outputdim] * len(listdim)).to(DEVICE), requires_grad=True)
        
        if self.Y is not None:
            bias=pt.nn.Parameter(self.Y.mean(0).clone().detach(), requires_grad= True)
        else: bias=pt.nn.Parameter(pt.tensor([[bias0] * outputdim], dtype=float).to(DEVICE), requires_grad=True)
        self.bias=bias
        
        self.optimizer = pt.optim.Adam(self.parameters(), lr=0.001)
        
        paramlist=[[params for params in mu.parameters()] for mu in self.list_W_mu] + [[params for params in alpha.parameters()] for alpha in self.list_W_logvar]
        for param in paramlist:
            self.optimizer.add_param_group({"params": param})

#----------------------------------------------------------------------------------------------------------------------------------------------
    # Forward function
    ## Input parameters :
    ##      self - object of class G2PSR
    ##      X - tensor containing genetic data formated per gene           
    def forward(self,X):
        # Encoding SNPs into the gene layer
        genarray= []
        # Set up the normal distribution parameters for the encoding layer with variational dropout parameter alpha.
        for i in range(len(X)):
            gen=pt.distributions.Normal(
                loc = self.list_W_mu[i](X[i].float()),
                scale = (self.alpha[i] + pt.log((self.list_W_mu[i](X[i].float()))**2 + 1e-8)).exp().pow(0.5)
            ) 
            genarray.append(gen)
        
        # Sample from the previously defined distribution and 
        # Concatenate the corresponding distribution into a single Tensor to obtain the gene layer.
        gensample=[]
        for g in genarray:
            gensample.append(g.rsample())
        gensample=pt.cat(gensample,1).float() # Concatenate all gensample Tensors : [sample;genes]
        
        # Decoding/Reconstructing physiological traits
        Y=pt.distributions.Normal(
            loc = (gensample @ self.mu) + self.bias,
            scale = self.noise.exp() # sigma parameter
            )
        
        ## Returns input, encoding SNPs to gene distribution, the gene layer and predicted output layer
        return {"X":X, "gene": genarray, "Z":gensample, "Y": Y}
    
#----------------------------------------------------------------------------------------------------------------------------------------------
    # Loss function
    ## Input parameters :
    ##      self - object of class G2PSR
    ##      pred - Forward function output Y (reconstructed phenotypic features) 
    ##      trueY - tensor containing the phenotypic data
    def loss_function(self, pred, trueY):
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        
        kl1=0
        kl2=0
        ll=0
        
        # Kullbarg-Leibler divergence for variational dropout
        kl1 -= ((k1 * pt.sigmoid(k2 + k3 * self.alpha)) - (0.5 * pt.log1p(self.alpha.exp().pow(-1)) - k1)).mean()
        # Log likelihood for variational encoding
        ll += pred.log_prob(trueY)
        
        # Avoid NaN values in the log likelihood function (decrepated ...)
        nanpos=ll!=ll
        if nanpos.sum()>0:
            trueloc=pred.loc==trueY
            prob_is_high=(nanpos * trueloc)
            prob_is_0=nanpos ^ prob_is_high
            ll[prob_is_high]=10
            ll[prob_is_0]=-1e3
        
        ll=ll.sum(1).mean(0)
        
        return (kl1 + kl2 - ll)
    
    
#----------------------------------------------------------------------------------------------------------------------------------------------
    # Probability Alpha function
    ## Function to set up sparsity alpha parameter
    ## Input parameters :
    ##      self - object of class G2PSR 
    def probalpha(self):
        alpha=self.alpha.exp()
        p=pt.mul(alpha, 1/(alpha+1))
        return p
    
    
#----------------------------------------------------------------------------------------------------------------------------------------------
    # Optimization function
    ## Input parameters :
    ##      self - object of class G2PSR  
    ##      X - tensor containing the genetic data formated per gene
    ##      Y - tensor containing phenotypic data
    ##      epochmax - total number of epoch to perform
    ##      step - epoch number when optimization parameters are stored
    ##      verbose - allow print of progress in the optimization progress
    def optimize(self, X = None, Y = None, epochmax = 10000, step=100, verbose = False):
        pt.cuda.empty_cache()
        if X is None: X=self.X
        if Y is None: Y=self.Y
        
        # parameters recorded during the optimization process
        losslist=[]
        plist=[]
        mulist=[]
        musnplist=[]
        ynoise=[]
        ybias=[]
        nb_snp=sum([el.shape[1] for el in self.X])
        
        for epoch in range(0, epochmax):
            pt.cuda.empty_cache()

            # Print progress in G2PSR optimization
            if (epoch * 100 % epochmax==0) and verbose:
                print(str(epoch * 100 / epochmax) + "%...")
                
                
            self.optimizer.zero_grad()  # Clears the gradients of all optimization
            
            # optimization
            pred=self.forward(X)
            loss=self.loss_function(pred["Y"], Y)
            loss.backward()
            
            # record optimization parameters of interest
            if (epoch % step == 0):
                # loss value
                losslist.append(loss)
                
                # alpha parameter per gene
                p=self.probalpha().detach().cpu().numpy() 
                plist.append(p)
                
                # mu parameter per gene
                mu=abs(self.mu.mean(1).detach().cpu().numpy())
                mulist.append(mu)
                
                # sigma parameter (reconstruction error)
                noise_value = self.noise.detach().tolist()[0]
                ynoise.append(noise_value)
                
                # biais parameter (reconstruction output)
                bias_value = self.bias.detach().tolist()
                ybias.append(bias_value)
                
                # mu parameter per gene
                tmp_mu_snp = [i.weight.detach().cpu().numpy().tolist()[0] for i in self.list_W_mu]
                musnp = [None]*nb_snp
                nb=0
                for idx, val in enumerate(tmp_mu_snp):
                    musnp[nb:nb+len(val)]= val
                    nb+=len(val)
                musnplist.append(musnp)
                
            self.optimizer.step()

        # Last optimization value storage
        losslist.append(loss)
        p=self.probalpha().detach().cpu().numpy()
        plist.append(p)
        mu=abs(self.mu.mean(1).detach().cpu().numpy())
        mulist.append(mu)
        noise_value = self.noise.detach().tolist()[0]
        ynoise.append(noise_value)
        tmp_mu_snp = [i.weight.detach().cpu().numpy().tolist()[0] for i in self.list_W_mu]
        musnp = [None]*nb_snp
        nb=0
        for idx, val in enumerate(tmp_mu_snp):
            musnp[nb:nb+len(val)]= val
            nb+=len(val)
        musnplist.append(musnp)
                
        pt.cuda.empty_cache()
        self.summary()
        self.early_stopping_strategy(losslist, ynoise, plist)
        # Return optimization parameter
        return ({"Losslist":losslist, "plist":plist, "mulist":mulist, 
                 "musnplist":musnplist, "ynoise":ynoise, "ybias":ybias})
    
    
    ## NEED TO REFINE THE SUMMARY FUNCTION AND PRINTABLE...
    def early_stopping_strategy(losslist, ynoise, plist, epochmax=1000, step=100):
        
        # Loss function plot
        loss = []
        epoch = []
        for i in range(0, len(losslist)):
            loss.append(float(losslist[i]))
            epoch.append(i*1000)
        loss_df = pd.DataFrame(data = {'Loss':loss, 'Epoch':epoch})
        
        (ggplot(loss_df, aes(x='Epoch', y='Loss'))
         + geom_line()
         + scale_y_log10()
         + theme_bw()
         + geom_vline(xintercept = 17000, color = "red")
         )
        
        
        
    
    
    def summary(self, ass_SNPs = False):
        if self.X is not None and self.Y is not None:
            if len(self.X)!=0 and len(self.Y)!=0:
                print("Average difference with target: " + str((self.forward(self.X)["Y"].rsample()-self.Y).mean()))
        
        print("probability that the gene is not relevant: ")
        prob=self.probalpha()
        i=0
        relevantgene=[]
        if (self.dfX is None):
            print(prob)
            return
        else:
            for i in range(0,len(self.dfX)):
                string=self.dfX[i][0] + ": " + "{:.4f}".format(prob[i].item())
                if (prob[i].item()<SNP_bnn._pmin): 
                    relevantgene.append(i)
                print(string)
                i+=1
            if len(relevantgene) != 0 :
                print("Gene(s) considered relevant:")
                for gene in relevantgene:
                    print(self.dfX[gene][0])
         
        if ass_SNPs == True :             
            for i in range(len(self.dfX)):
                print_gene = 0
                for i2 in range(self.list_W_mu[i].in_features):
                    strmu=""
                    if abs(self.list_W_mu[i].weight[0][i2]) >= 1:
                        if print_gene == 0 :
                            print("Most important SNP(s) of gene " + self.dfX[i][0] + ":")
                            print_gene += 1
                        strmu+=str(round(self.list_W_mu[i].weight[0][i2].item(),3)) 
                        string="\t" + self.dfX[i][1].columns[i2] + ": mu=" + strmu
                        print(string)
        
        
        

        
        
    @classmethod
    def gfptoCSV(cls, samplesize, nb_gene, nb_trait, nb_W2dim, pct_trait, filename = "", noise=float(0.05), save_data = False):
        nbSNPperGene = pd.read_csv('/home/mdeprez/Documents/Data_ADNI/pathways-bnn/nbSNPperGene.csv', names=("Gene", "Nb_snps"))
        
        X=[]
        X_csv=[]
        X_Gsnp=[]
        # number of SNPs per gene
        nb_SNP = np.random.choice(nbSNPperGene.loc[:,"Nb_snps"].to_numpy(), size = nb_gene, replace=False)
        for i in range(0,nb_gene):
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
            # Adapt the number of target dimension.
            Wi=abs(np.random.randn(X[i].shape[1],nb_trait) * 5)
            if X[i].shape[1] > 5 :
                nb_snp_use = np.random.randint(5, X[i].shape[1], 1)
                Wi[-(X[i].shape[1]-nb_snp_use[0]):,:]=0                        
            trait_ind = int(np.floor(nb_trait*(pct_trait/100)))
            Wi[:, trait_ind:] = Wi[:, trait_ind:] / 5
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
        noise_std = np.std(Y[:,0].numpy()) * noise
        Y=Y + pt.tensor([[cls._bias] * Y.shape[1]] * Y.shape[0], device=DEVICE)
        noise_t=pt.normal(mean=pt.tensor([[0] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE), 
                    std=pt.tensor([[noise_std * cls._bias] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE))
        Y_noise=Y + noise_t
        pt.cuda.empty_cache()
        
        if save_data: 
            complete_filename=filename+"%is_%ig_%it_%ipt_%itg_%.2fn"%(samplesize, nb_gene, nb_trait, pct_trait, nb_W2dim, noise)
            if not os.path.exists(complete_filename):
                os.mkdir(complete_filename)
                np.savetxt(complete_filename+"/gen_matrix.csv", np.vstack(X_csv), delimiter=";")
                np.savetxt(complete_filename+"/gen_snp.csv", np.vstack(X_Gsnp), delimiter=";")
                np.savetxt(complete_filename+"/w_snp_target.csv", np.vstack(W_csv), delimiter=";")
                np.savetxt(complete_filename+"/target.csv", Y.numpy().transpose(), delimiter=";")
                np.savetxt(complete_filename+"/target_noise.csv", Y_noise.numpy().transpose(), delimiter=";")
        return ({"X":X, "W":W, "Z":Z, "Y":Y, "X_csv":X_csv, "X_Gsnp":X_Gsnp, "W_csv":W_csv})
    
    

    # def loadGeneticData(linkGenotype = _defaultGL, linkSNPmap= _defaultmap):
    #     gencsv = pd.read_csv(linkGenotype, header=0, index_col=0)
        
    #     # we deal with missing values (-1 in the csv)
    #     gencsv = gencsv[gencsv != -1]
    #     mean = gencsv.mean(axis=1)
    #     gencsv=gencsv.transpose() #fillna only works column by colum with series, so we transpose. We would need to transpose later anyway
    #     gencsv.fillna(mean,inplace=True)
        
    #     # switch the 0 values into 2 for alternative homozygote and 2 values into 0 for reference homozygote
    #     gencsv[gencsv == 2] = 4
    #     gencsv[gencsv == 0] = 2
    #     gencsv[gencsv == 4] = 0
        
    #     genmap = pd.read_csv(linkSNPmap, header=0, index_col=0)
    #     return {"genotype":gencsv, "SNP into genes":genmap}
    
    # def loadVolumetricData(linkVolume = _defaultVL, linkCognition = _defaultCL):
    #     volumes = pd.read_csv(linkVolume, header=0, index_col=1).drop("Unnamed: 0", axis=1)
    #     cognition = pd.read_csv(linkCognition, header=0, index_col=1).drop("Unnamed: 0", axis=1)
    #     return {"Volume":volumes, "cognition":cognition}
    
    # @classmethod
    # def loadData(cls, linkGenotype = _defaultGL, linkSNPmap= _defaultmap, linkVolume = _defaultVL, linkCognition = _defaultCL, verbose=False):
    #     #loading the data
    #     if verbose: print("Loading Genetic Data")
    #     temp=cls.loadGeneticData(linkGenotype, linkSNPmap)
    #     genotype=temp["genotype"]
    #     genmap=temp["SNP into genes"]
    #     if verbose: print("Loading Phenotypic Data")
    #     temp=cls.loadVolumetricData(linkVolume, linkCognition)
    #     volumetric_data=temp["Volume"]
    #     cognition_data=temp["cognition"]
        
    #     # Define the number of physiological dimension to take into account
    #     physio_dim=-volumetric_data.shape[1]-cognition_data.shape[1]
        
    #     # Reshape data
    #     if verbose: print("Redefine data containers ...")
    #     ## Create the correspondance between sample names
    #     ## We take the last 4 digits and cast them as int to get the RID in the same type as for the other dataframe
    #     genotype.rename(index = lambda s: int(s[-4:]), inplace=True) 
    #     ## Concatenate all data (physiological and genetic)
    #     physio_data=pd.concat([volumetric_data,cognition_data], axis=1, join="inner")
    #     data=pd.concat([genotype,physio_data], axis=1, join="inner")
        
    #     # Define the output layer of the neural network as the physiological data
    #     Y=data.iloc[:,physio_dim:]
    #     Y=Y[["Hippocampus.bl","Entorhinal.bl","CDRSB.bl","ADAS11.bl","MMSE.bl","RAVLT.immediate.bl","RAVLT.forgetting.bl","FAQ.bl"]]
    #     ## Create the corresponding tensor
    #     tensorY=pt.tensor(Y.values, device=DEVICE, dtype=float)
        
    #     # Define the input layer of the neural network as the genetic data
    #     X=data.iloc[:,:physio_dim]
    #     # Divide the genetic data by their associated gene : a list of genetic matrices (x SNPs) by gene
    #     Xlist={}
    #     for gene in genmap.columns:
    #         Xi=pd.DataFrame()
    #         for snp in X.columns:
    #             if(genmap[gene][snp]==1):
    #                     Xi=pd.concat([Xi,X[snp]], axis=1)
    #         Xlist[gene]=Xi
    #     # Create the corresponding tensor
    #     tensorX=[]
    #     for gene in Xlist:
    #         tensorX.append(pt.tensor(Xlist[gene].values, device=DEVICE, dtype=float))
    #     return {"data":data, "X":Xlist, "Tensor X":tensorX, "Y":Y, "Tensor Y":tensorY}
    
    
    @classmethod
    def boxplot(cls,snplist,featlist):
        data=cls.loadData()["data"]
        for i in range(len(snplist)):
            rssnp=snplist[i]
            for i2 in range(len(featlist)):
                vol=featlist[i2]
                temp=data[[rssnp,vol]]
                fig,ax=plt.subplots()
                ax.set_title(rssnp+" "+vol)
                ax.boxplot([temp[temp[rssnp] == 0][vol], 
                            temp[temp[rssnp] == 1][vol], 
                            temp[temp[rssnp] == 2][vol]],
                            labels=[0,1,2])
        plt.close()

