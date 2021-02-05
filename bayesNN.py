#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:23:17 2020

@author: mdeprez
"""

import torch as pt
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from gpu import DEVICE # Check if GPU core available
import os

# DEVICE = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

class SNP_bnn(pt.nn.Module):
    strand_max=200
    strand_min=50

    _defaultVL = "https://marcolorenzi.github.io/material/winter_school/volumes.csv"
    _defaultCL = "https://marcolorenzi.github.io/material/winter_school/cognition.csv"
    _defaultGL = "/user/mdeprez/home/Documents/Data_ADNI/Plink_LD/Genotype_matrice_500G.txt"
    _defaultmap = "/user/mdeprez/home/Documents/Data_ADNI/Plink_LD/matrix_snp_gene_500G.txt"
    _pmin= 0.05     # the value of p under which we keep the gene for analysis of its SNP
    _bias=1         # the average value of the generated physiological traits

    
    # Initialisation function
    ## Input parameters :
    ##      self - object of class SNP_bnn
    ##      input_list - tensor containing genetic data         | Obtained in functions CSVtoBayesNN ()
    ##      output_shape - tensor containing physiologic data   |                   and genfullprofile()
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
        
        # Creating noise parameter for decoding layer estimate
        self.noise = pt.nn.Parameter(pt.Tensor([[noise0] * output_shape[1]]), requires_grad=True) # dimension of the output
        
        # Creating the parameters in the decoding part (dropout with log(alpha), mu' and the bias associated with the physiological traits)
        self.alpha=pt.nn.Parameter(pt.Tensor([[alpha0]] * len(listdim)), requires_grad=True) #alpha is a log
        self.mu=pt.nn.Parameter(pt.Tensor([[mu0] * outputdim] * len(listdim)), requires_grad=True)
        
        if self.Y is not None:
            bias=pt.nn.Parameter(self.Y.mean(0).clone().detach(), requires_grad=True)
        else: bias=pt.nn.Parameter(pt.tensor([[bias0] * outputdim], dtype=float), requires_grad=True)
        self.bias=bias
        
        self.optimizer = pt.optim.Adam(self.parameters(), lr=0.001)
        
        paramlist=[[params for params in mu.parameters()] for mu in self.list_W_mu] + [[params for params in alpha.parameters()] for alpha in self.list_W_logvar]
        for param in paramlist:
            self.optimizer.add_param_group({"params": param})

    # Format input data into the appropriate neural network-compatible format     
    ## linkGenotype - Genotype matrix (SNPs as rows and samples as columns)
    ## linkSNPmap - SNP-to-Gene matrix, correspondance between SNPs and their related genes (0 or 1 values, SNPs as rows and Genes as columns)
    ## linkVolume - Phenotype data, brain volumes information
    ## linkCognition - Phenotype data, memory test metric
    @classmethod
    def CSVtoBayesNN(cls, linkGenotype=_defaultGL, linkSNPmap=_defaultmap, linkVolume=_defaultVL, linkCognition=_defaultCL, verbose=False):
        data=cls.loadData(linkGenotype,linkSNPmap, linkVolume, linkCognition)
        bayesNN=cls(data["Tensor X"], data["Tensor Y"]).to(DEVICE)
        bayesNN.dfX=[]
        bayesNN.dfY=[]
        i=0
        for gene in data["X"].keys():
            bayesNN.dfX.append([gene, data["X"][gene], bayesNN.probalpha()[i]])
            i += 1
        if verbose : print("Initialized neural network ready.")
        return bayesNN
            
    def forward(self,X):
        # Encoding SNPs into the gene layer
        genarray= []
        # Set up the normal distribution parameters for the encoding layer
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
        
        # Decoding into physiological traits
        # need Y as a normal distribution for loglikelihood
        Y=pt.distributions.Normal(
            loc = (gensample @ self.mu) + self.bias,
            scale = self.noise.exp()
            )
        
        ## Returns input, encoding SNPs to gene distribution, the gene layer and predicted output layer
        return {"X":X, "gene": genarray, "Z":gensample, "Y": Y}
    
    
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
        
        # Avoid NaN values in the log likelihood function (decrepated ?)
        nanpos=ll!=ll
        if nanpos.sum()>0:
            trueloc=pred.loc==trueY
            prob_is_high=(nanpos * trueloc)
            prob_is_0=nanpos ^ prob_is_high
            ll[prob_is_high]=10
            ll[prob_is_0]=-1e3
        
        ll=ll.sum(1).mean(0)
        
        return (kl1 + kl2 - ll)
    
    def probalpha(self):
        alpha=self.alpha.exp()
        p=pt.mul(alpha, 1/(alpha+1))
        return p
    
    def optimize(self, X = None, Y = None, epochmax = 10000, step=100, verbose = False):
        pt.cuda.empty_cache()
        if X is None: X=self.X
        if Y is None: Y=self.Y
        losslist=[]
        plist=[]
        mulist=[]
        for epoch in range(0, epochmax):
            pt.cuda.empty_cache()

            # Print progress in VAE optimization
            if (epoch * 100 % epochmax==0) and verbose:
                print(str(epoch * 100 / epochmax) + "%...")
                
            self.optimizer.zero_grad()  # Clears the gradients of all optimization
            
            # Compare decoded output with the real ones
            pred=self.forward(X)
            loss=self.loss_function(pred["Y"], Y)
            loss.backward()
            
            if (epoch % step == 0):
                losslist.append(loss)
                p=self.probalpha().detach().cpu().numpy() #add the probability of the genes being not relevant
                plist.append(p)
                mu=abs(self.mu.mean(1).detach().cpu().numpy()) #add the mean of the weights
                mulist.append(mu)

            self.optimizer.step()

            
        ## Plot making -------------------------------------------------------------------------------
        # we add a final value for the plots to be complete even if epochmax is not a multiple of step
        losslist.append(loss)
        p=self.probalpha().detach().cpu().numpy()
        plist.append(p)
        mu=abs(self.mu.mean(1).detach().cpu().numpy())
        mulist.append(mu)
        
        # Plot making --  NEEDS TO BE REFINED
        indexlist=list(range(0,epochmax,step))
        indexlist.append(epochmax)
        fig=plt.figure()
        plt.plot(indexlist[1:],losslist[1:], figure=fig)
        fig.suptitle("Loss function depending on epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss value")
        fig2=plt.figure()
        plist=np.reshape(plist, (len(plist), len(plist[0]))).transpose()
        if (len(X)<11):
            for i in range(len(plist)):
                if self.dfX is None:
                    plt.plot(indexlist,plist[i], figure=fig2, label=i)
                else:
                    plt.plot(indexlist,plist[i], figure=fig2, label=self.dfX[i][0])
        else:
            for i in range(1):
                if self.dfX is None:
                    plt.plot(indexlist,plist[i], 'o', figure=fig2, label=i)
                else:
                    plt.plot(indexlist,plist[i], 'o', figure=fig2, label=self.dfX[i][0])
            for i in range(1, len(plist)):
                if self.dfX is None:
                    plt.plot(indexlist,plist[i], figure=fig2)
                else:
                    plt.plot(indexlist,plist[i], figure=fig2)
        plt.plot(indexlist ,[SNP_bnn._pmin] * len(indexlist), '--k', figure=fig2, label="p=0.05")
        plt.legend()
        fig2.suptitle("probability of the dimension being non significant")
        plt.xlabel("epoch")
        plt.ylabel("p(alpha)")
        plt.ylim(0,1)
        fig3=plt.figure()
        mulist=np.reshape(mulist, (len(mulist), len(mulist[0]))).transpose()
        for i in range(len(mulist)):
            if self.dfX is None:
                plt.plot(indexlist,mulist[i], figure=fig3, label=i)
            else:
                plt.plot(indexlist,mulist[i], figure=fig3, label=self.dfX[i][0])
        fig3.suptitle("Weight depending on epoch")
        plt.xlabel("epoch")
        plt.ylabel("weight")
        plt.legend()
        plt.ylim(bottom=0)
        
        pt.cuda.empty_cache()
        self.summary()
        return ({"Losslist":losslist, "plist":plist, "mulist":mulist})
    
    
    ## NEED TO REFINE THE SUMMARY FUNCTION AND PRINTABLE...
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
    def genfullprofile(cls, samplesize, nb_gene, nb_trait, nb_W2dim, noise = float(0.05)):
        X=[]
        for i in range(0,nb_gene):
            nb_SNP=np.random.randint(cls.strand_min,cls.strand_max)
            SNP=np.floor(abs(np.random.randn(samplesize, nb_SNP)))
            for sample in range(0,SNP.shape[0]):
                for snp in range(0,SNP.shape[1]):
                    if SNP[sample][snp]>2: 
                        SNP[sample][snp]=2
            SNP = pt.tensor(SNP, device=DEVICE, dtype=float)
            X.append(SNP)
        W=[]
        Z=[]
        for i in range(nb_W2dim):
            Wi=abs(np.random.randn(X[i].shape[1],nb_trait))
            for i2 in range(4, nb_trait):
                Wi[:,i2]=0
            Wi=pt.tensor(Wi, device=DEVICE, dtype=float)
            W.append(Wi)
            Z.append(X[i] @ Wi)
        for i in range(nb_W2dim, nb_gene):
            Wi=[[0] * nb_trait] * X[i].shape[1]
            Wi=pt.tensor(Wi, device=DEVICE, dtype=float)
            W.append(Wi)
            Z.append(X[i] @ Wi)
        Y=sum(Z)
        Y=Y + pt.tensor([[cls._bias] * Y.shape[1]] * Y.shape[0], device=DEVICE)
        noise=pt.normal(mean=pt.tensor([[0] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE), 
                    std=pt.tensor([[noise * cls._bias] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE))
        Y=Y + noise
        pt.cuda.empty_cache()
        return ({"X":X, "W":W, "Z":Z, "Y":Y})
    
    # redined genfullprofile function to save synthetic dataset.
    @classmethod
    def gfptoCSV(cls, samplesize, nb_gene, nb_trait, nb_W2dim, filename, noise=float(0.05) ):
        complete_filename=filename+"%is_%ig_%it_%itg_%.2fn"%(samplesize, nb_gene, nb_trait, nb_W2dim, noise)
        
        X=[]
        X_csv=[]
        X_Gsnp=[]
        for i in range(0,nb_gene):
            nb_SNP=np.random.randint(cls.strand_min,cls.strand_max)
            SNP=np.floor(abs(np.random.randn(samplesize, nb_SNP)))
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
            nb_snp_use = np.random.random_integers(3,X[i].shape[1], 1)
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
       # Y=Y + pt.tensor([[cls._bias] * Y.shape[1]] * Y.shape[0], device=DEVICE)
        noise=pt.normal(mean=pt.tensor([[0] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE), 
                    std=pt.tensor([[noise * cls._bias] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE))
        Y=Y + noise
        pt.cuda.empty_cache()
        
        if not os.path.exists(complete_filename):
            os.mkdir(complete_filename)
            np.savetxt(complete_filename+"/gen_matrix.csv", np.vstack(X_csv), delimiter=";")
            np.savetxt(complete_filename+"/gen_snp.csv", np.vstack(X_Gsnp), delimiter=";")
            np.savetxt(complete_filename+"/w_snp_target.csv", np.vstack(W_csv), delimiter=";")
            np.savetxt(complete_filename+"/target.csv", Y.numpy().transpose(), delimiter=";")
        return ({"X":X, "W":W, "Z":Z, "Y":Y, "X_csv":X_csv, "X_Gsnp":X_Gsnp, "W_csv":W_csv})
    
    

    def loadGeneticData(linkGenotype = _defaultGL, linkSNPmap= _defaultmap):
        gencsv = pd.read_csv(linkGenotype, header=0, index_col=0)
        
        # we deal with missing values (-1 in the csv)
        gencsv = gencsv[gencsv != -1]
        mean = gencsv.mean(axis=1)
        gencsv=gencsv.transpose() #fillna only works column by colum with series, so we transpose. We would need to transpose later anyway
        gencsv.fillna(mean,inplace=True)
        
        # switch the 0 values into 2 for alternative homozygote and 2 values into 0 for reference homozygote
        gencsv[gencsv == 2] = 4
        gencsv[gencsv == 0] = 2
        gencsv[gencsv == 4] = 0
        
        genmap = pd.read_csv(linkSNPmap, header=0, index_col=0)
        return {"genotype":gencsv, "SNP into genes":genmap}
    
    def loadVolumetricData(linkVolume = _defaultVL, linkCognition = _defaultCL):
        volumes = pd.read_csv(linkVolume, header=0, index_col=1).drop("Unnamed: 0", axis=1)
        cognition = pd.read_csv(linkCognition, header=0, index_col=1).drop("Unnamed: 0", axis=1)
        return {"Volume":volumes, "cognition":cognition}
    
    @classmethod
    def loadData(cls, linkGenotype = _defaultGL, linkSNPmap= _defaultmap, linkVolume = _defaultVL, linkCognition = _defaultCL, verbose=False):
        #loading the data
        if verbose: print("Loading Genetic Data")
        temp=cls.loadGeneticData(linkGenotype, linkSNPmap)
        genotype=temp["genotype"]
        genmap=temp["SNP into genes"]
        if verbose: print("Loading Phenotypic Data")
        temp=cls.loadVolumetricData(linkVolume, linkCognition)
        volumetric_data=temp["Volume"]
        cognition_data=temp["cognition"]
        
        # Define the number of physiological dimension to take into account
        physio_dim=-volumetric_data.shape[1]-cognition_data.shape[1]
        
        # Reshape data
        if verbose: print("Redefine data containers ...")
        ## Create the correspondance between sample names
        ## We take the last 4 digits and cast them as int to get the RID in the same type as for the other dataframe
        genotype.rename(index = lambda s: int(s[-4:]), inplace=True) 
        ## Concatenate all data (physiological and genetic)
        physio_data=pd.concat([volumetric_data,cognition_data], axis=1, join="inner")
        data=pd.concat([genotype,physio_data], axis=1, join="inner")
        
        # Define the output layer of the neural network as the physiological data
        Y=data.iloc[:,physio_dim:]
        Y=Y[["Hippocampus.bl","Entorhinal.bl","CDRSB.bl","ADAS11.bl","MMSE.bl","RAVLT.immediate.bl","RAVLT.forgetting.bl","FAQ.bl"]]
        ## Create the corresponding tensor
        tensorY=pt.tensor(Y.values, device=DEVICE, dtype=float)
        
        # Define the input layer of the neural network as the genetic data
        X=data.iloc[:,:physio_dim]
        # Divide the genetic data by their associated gene : a list of genetic matrices (x SNPs) by gene
        Xlist={}
        for gene in genmap.columns:
            Xi=pd.DataFrame()
            for snp in X.columns:
                if(genmap[gene][snp]==1):
                        Xi=pd.concat([Xi,X[snp]], axis=1)
            Xlist[gene]=Xi
        # Create the corresponding tensor
        tensorX=[]
        for gene in Xlist:
            tensorX.append(pt.tensor(Xlist[gene].values, device=DEVICE, dtype=float))
        return {"data":data, "X":Xlist, "Tensor X":tensorX, "Y":Y, "Tensor Y":tensorY}
    
    
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

