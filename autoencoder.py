# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:01:39 2020

@author: morei
"""


import torch as pt
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

DEVICE = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
        
class Autoencoder(pt.nn.Module):
    
    def __init__(self, input_shape, latentdim):
        super().__init__()
        if type(input_shape) is pt.Size:
            inputdim=input_shape[1]
        elif type(int(input_shape)) is int:
            inputdim=input_shape
        self.W_mu=pt.nn.Linear(inputdim, latentdim)
        self.W_logvar=pt.nn.Linear(inputdim, latentdim)
        self.W_mu_out=pt.nn.Linear(latentdim, inputdim)
        self.W_logvar_out=pt.nn.Linear(latentdim, inputdim)
        self.optimizer = pt.optim.Adam(self.parameters())

    def encode(self, X):
        Z = pt.distributions.Normal(
            loc=self.W_mu(X),
            scale=self.W_logvar(X).exp().pow(0.5) #<- here change variance for wmu * alpha²
		)
        return Z
        

    def sample(self, Z):
        if self.training:
            return Z.rsample()
        else:
            return Z.loc

    def decode(self, Z2):
        X2 = pt.distributions.Normal(
            loc=self.W_mu_out(Z2),
            scale=self.W_logvar_out(Z2).exp().pow(0.5)
		)
        return X2
    
    def forward(self,X):
        Z=self.encode(X)
        Z2=self.sample(Z)
        X2=self.decode(Z2)
        return {"X": X, "Z": Z, "X'":X2}
    
    def loss_function(fwd_return):
        X = fwd_return['X']
        Z = fwd_return['Z']
        X2 = fwd_return["X'"]
  
        kl = 0
        ll = 0
  
  			# KL Divergence
        kl += pt.distributions.kl_divergence(Z, pt.distributions.Normal(0, 1)).sum(1).mean(0)  # torch.Size([1])
        ll += X2.log_prob(X).sum(1).mean(0)
  
        total = kl - ll
  
        losses = {
  			'total': total,
  			'kl': kl,
  			'll': ll
  		}
        
        return losses

    
    def optimize(self,X, epochmax):
        losslist=[]
        for epoch in range(0, epochmax):
            self.optimizer.zero_grad()
            pred=self.forward(X)
            loss=Autoencoder.loss_function(pred)['total']
            loss.backward()
            losslist.append(loss.item())
            self.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        print(pred)
        return self.state_dict()
    
class Autoencoder_multiplelayers(pt.nn.Module):
    
    def __init__(self, input_shape, dimgene, dimpathways):
        super().__init__()
        self.outerAE=Autoencoder(input_shape, dimgene)
        self.innerAE=Autoencoder(dimgene, dimpathways)
        
        self.W_mu=pt.nn.Linear(input_shape[1], dimpathways)
        self.W_logvar=pt.nn.Linear(input_shape[1], dimpathways)
        self.W_mu_out=pt.nn.Linear(dimpathways, input_shape[1])
        self.W_logvar_out=pt.nn.Linear(dimpathways, input_shape[1])
        
        self.optimizer = pt.optim.Adam(self.parameters())
        
    def encode(self, X):
        Z = pt.distributions.Normal(
            loc=self.W_mu(X),
            scale=self.W_logvar(X).exp().pow(0.5)
		)
        return Z
        

    def sample(self, Z):
        if self.training:
            return Z.rsample()
        else:
            return Z.loc

    def decode(self, Z2):
        X2 = pt.distributions.Normal(
            loc=self.W_mu_out(Z2),
            scale=self.W_logvar_out(Z2).exp().pow(0.5)
		)
        return X2
    
    def forward(self,X):
        Z=self.encode(X)
        Z2=self.sample(Z)
        X2=self.decode(Z2)
        return {"X": X, "Z": Z, "X'":X2}
    
    def loss_function(fwd_return):
        X = fwd_return['X']
        Z = fwd_return['Z']
        X2 = fwd_return["X'"]
  
        kl = 0
        ll = 0
  
  			# KL Divergence
        kl += pt.distributions.kl_divergence(Z, pt.distributions.Normal(0, 1)).sum(1).mean(0)  # torch.Size([1])
        ll += X2.log_prob(X).sum(1).mean(0)
  
        total = kl - ll
  
        losses = {
  			'total': total,
  			'kl': kl,
  			'll': ll
  		}
        
        return losses

    
    def optimize(self,X, epochmax, mode=0):
        if mode==0:
            self.optimize(X, epochmax, mode=1)
            self.optimize(self.outerAE.encode(X).rsample(), epochmax, mode=2)
            currentAE=self
        elif mode==1:
            currentAE=self.outerAE
        elif mode==2:
            currentAE=self.innerAE
        else:
            print("wrong mode")
            return
        losslist=[]
        for epoch in range(0, epochmax):
            currentAE.optimizer.zero_grad()
            pred=currentAE.forward(X)
            loss=Autoencoder.loss_function(pred)['total']
            loss.backward(retain_graph=True)
            losslist.append(loss.item())
            currentAE.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        print(pred)
        return self.state_dict()
    
class VDAutoencoder(pt.nn.Module):
    
    def __init__(self, input_shape, latentdim):
        super().__init__()
        if type(input_shape) is pt.Size:
            inputdim=input_shape[1]
        elif type(int(input_shape)) is int:
            inputdim=input_shape
        self.W_mu=pt.nn.Linear(inputdim, latentdim)
        self.W_logvar=pt.nn.Linear(inputdim, latentdim)
        self.W_mu_out=pt.nn.Linear(latentdim, inputdim)
        self.W_logvar_out=pt.nn.Linear(latentdim, inputdim)
        self.latentdim=latentdim
        self.alpha=pt.Tensor([0.5] * latentdim)
        self.alpha.requires_grad=True
        self.optimizer = pt.optim.Adam(self.parameters())
        self.optimizer.add_param_group({ "params":self.alpha})

    def encode(self, X):
        Z = pt.distributions.Normal(
            loc=self.W_mu(X),
            scale=(self.alpha + pt.log(self.W_mu(X)**2 + 1e-8)).exp().pow(0.5) #self.W_mu**2 + 1e-8 can be replaced by pt.abs(self.W_mu + 1e-8)
		)
        return Z
        

    def sample(self, Z):
        if self.training:
            return Z.rsample()
        else:
            return Z.loc

    def decode(self, Z2):
        X2 = pt.distributions.Normal(
            loc=self.W_mu_out(Z2),
            scale=self.W_logvar_out(Z2).exp().pow(0.5)
		)
        return X2
    
    def forward(self,X):
        Z=self.encode(X)
        Z2=self.sample(Z)
        X2=self.decode(Z2)
        return {"X": X, "Z": Z, "X'":X2}
    
    def loss_function(self, fwd_return):
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        X = fwd_return['X']
        X2 = fwd_return["X'"]
  
        kl = 0
        ll = 0
  
  			# KL Divergence
        kl -= (k1 * pt.sigmoid(k2 + k3 * self.alpha) - 0.5 * pt.log1p(self.alpha.exp().pow(-1)) - k1).mean(0)
        ll += X2.log_prob(X).sum(1).mean(0)
  
        total = kl - ll
  
        losses = {
  			'total': total,
  			'kl': kl,
  			'll': ll
  		}
        
        return losses

    
    def optimize(self,X, epochmax):
        losslist=[]
        alpha1=[]
        alpha2=[]
        for epoch in range(0, epochmax):
            if (self.alpha[0] != self.alpha[0]):
                print(losslist[len(losslist)-1])
                break
            self.optimizer.zero_grad()
            pred=self.forward(X)
            loss=self.loss_function(pred)['total'].mean()
            loss.backward(retain_graph=True)
            losslist.append(loss)
            alpha1.append(self.alpha[0].item())
            alpha2.append(self.alpha[1].item())
            self.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        fig2=plt.figure()
        plt.plot(alpha1, figure=fig2)
        plt.plot(alpha2, figure=fig2)
        print(pred["Z"].rsample())
        return self.state_dict()
    
class VDonWeightAE(pt.nn.Module): #seems extremely robust to noise when it comes to determining relevant dimensions
    
    def __init__(self, input_shape, output_shape, mu0=0.5, alpha0=0.5):
        super().__init__()
        if type(input_shape) is pt.Tensor:
            input_shape=input_shape.shape
        if type(output_shape) is pt.Tensor:
            output_shape=output_shape.shape
        if type(input_shape) is pt.Size or type(input_shape) is list:
            inputdim=input_shape[1]
        else: return("error")
        if type(output_shape) is pt.Size or type(output_shape) is list:
            outputdim=output_shape[1]
        elif type(output_shape) is int:
            outputdim=output_shape
        else: return("error")
        self.mu=pt.nn.Parameter(pt.Tensor([[mu0] * outputdim] * inputdim), requires_grad=True)
        self.alpha=pt.nn.Parameter(pt.Tensor([[0.5]] * inputdim), requires_grad=True) #alpha is a log to avoid getting nan
        self.optimizer = pt.optim.Adam(self.parameters())
        
    def probalpha(self):
        alpha=self.alpha.exp()
        p=pt.mul(alpha, 1/(alpha+1))
        return p
        
    def encode(self, X):
        pW=pt.distributions.Normal(self.mu,(self.alpha + pt.log(self.mu**2 + 1e-8)).exp().pow(0.5))
        W=pW.rsample()
        Y=X@W
        return Y
    
    def loss_function(self, trueY, pred):
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        kl = (k1 * pt.sigmoid(k2 + k3 * self.alpha) - 0.5 * pt.log1p(self.alpha.exp().pow(-1)) - k1).mean()
        cost = ((pred - trueY)**2).mean()
        return (cost-kl)
    
    def optimize(self,X, Y, epochmax):
        losslist=[]
        for epoch in range(0, epochmax):
            self.optimizer.zero_grad()
            pred=self.encode(X)
            loss=self.loss_function(Y, pred)
            loss.backward(retain_graph=True)
            losslist.append(loss)
            self.optimizer.step()
        fig=plt.figure()
        plt.plot(losslist, figure=fig)
        print(pred)
        return self.state_dict()
        
    
class SNPAutoencoder(pt.nn.Module):
    strand_max=200
    strand_min=50
    
    _defaultVL = "https://marcolorenzi.github.io/material/winter_school/volumes.csv"
    _defaultCL = "https://marcolorenzi.github.io/material/winter_school/cognition.csv"
    _defaultGL = "Genotype_matrix_example1.csv"
    _defaultmap = "matrix_snp_gene_example_1.csv"
    _pmin= 0.05 #the value of p under which we keep the gene for analysis of its SNP
    
    def __init__(self, input_list, output_shape, mu0=1, alpha0=0):
        super().__init__()
        
        #getting the dimension
        listdim=[]
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
        
        #creating the parameters
        list_W_mu=[]
        list_W_logvar=[]
        for i in range(0, len(listdim)):
            list_W_mu.append(pt.nn.Linear(listdim[i], 1, bias=False).to(DEVICE))
            list_W_logvar.append(pt.nn.Linear(listdim[i],1, bias=False).to(DEVICE))
        self.list_W_mu=list_W_mu
        self.list_W_logvar=list_W_logvar
        self.alpha=pt.nn.Parameter(pt.Tensor([[alpha0]] * len(listdim)), requires_grad=True) #alpha is a log
        self.mu=pt.nn.Parameter(pt.Tensor([[mu0] * outputdim] * len(listdim)), requires_grad=True)
        self.optimizer = pt.optim.Adam(self.parameters(), lr=0.001)
        paramlist=[[params for params in mu.parameters()] for mu in self.list_W_mu] + [[params for params in alpha.parameters()] for alpha in self.list_W_logvar]
        for param in paramlist:
            self.optimizer.add_param_group({"params": param})
            
    @classmethod
    def CSVtoAutoEncoder(cls, linkGenotype=_defaultGL, linkSNPmap=_defaultmap, linkVolume=_defaultVL, linkCognition=_defaultCL):
        data=cls.loadData(linkGenotype,linkSNPmap, linkVolume, linkCognition)
        autoencoder=cls(data["Tensor X"], data["Tensor Y"]).to(DEVICE)
        autoencoder.dfX=[]
        autoencoder.dfY=[]
        i=0
        for gene in data["X"].keys():
            autoencoder.dfX.append([gene, data["X"][gene], autoencoder.probalpha()[i]])
            i += 1
        return autoencoder
            
    def forward(self,X):
        #encoding into genes
        genarray= []
        for i in range(len(X)):
            gen=pt.distributions.Normal(
                loc = self.list_W_mu[i](X[i].float()),
                scale = (self.list_W_logvar[i](X[i].float())).exp().pow(0.5)
            )
            genarray.append(gen)
        
        #encoding into physiological traits
        gensample=[]
        for g in genarray:
            gensample.append(g.rsample())
        gensample=pt.cat(gensample,1).float()
        
        #need Y as a normal distribution for loglikelyhood
        Y=pt.distributions.Normal(
            loc = gensample @ self.mu,
            scale = abs(gensample) @ (self.alpha + pt.log(self.mu**2 + 1e-8)).exp().pow(0.5)
            )
        
        return {"X":X, "gene": genarray, "Z":gensample, "Y": Y}
    
    def loss_function(self, pred, trueY):
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        
        kl1=0
        kl2=0
        ll=0
        
        kl1 -= (k1 * pt.sigmoid(k2 + k3 * self.alpha) - 0.5 * pt.log1p(self.alpha.exp().pow(-1)) - k1).mean()
        # for i in range(len(self.list_W_alpha)):
        #     kl2 -= (k1 * pt.sigmoid(k2 + k3 * self.list_W_alpha[i]) - 0.5 * pt.log1p(self.list_W_alpha[i].exp().pow(-1)) - k1).mean()
        ll += pred.log_prob(trueY).sum(1).mean(0)
        return (kl1 + kl2 - ll)
    
    def probalpha(self):
        alpha=self.alpha.exp()
        p=pt.mul(alpha, 1/(alpha+1))
        return p
    
    def optimize(self, X = None, Y = None, epochmax = 10000, step=100):
        pt.cuda.empty_cache()
        if X is None: X=self.X
        if Y is None: Y=self.Y
        losslist=[]
        plist=[]
        mulist=[]
        for epoch in range(0, epochmax):
            pt.cuda.empty_cache()
            if (epoch * 100 % epochmax==0):
                print(str(epoch * 100 / epochmax) + "%...")
            self.optimizer.zero_grad()
            pred=self.forward(X)
            loss=self.loss_function(pred["Y"], Y)
            loss.backward(retain_graph=True)
            if (epoch % step == 0):
                losslist.append(loss)
                p=self.probalpha().detach().cpu().numpy() #add the probability of the genes being not relevant
                plist.append(p)
                mu=abs(self.mu.mean(1).detach().cpu().numpy()) #add the mean of the weights
                mulist.append(mu)
            self.optimizer.step()
            
        #we add a final value for the plots to be complete even if epochmax is not a multiple of step
        losslist.append(loss)
        p=self.probalpha().detach().cpu().numpy()
        plist.append(p)
        mu=abs(self.mu.mean(1).detach().cpu().numpy())
        mulist.append(mu)
        
        #we make the plots
        indexlist=list(range(0,epochmax,step))
        indexlist.append(epochmax)
        fig=plt.figure()
        plt.plot(indexlist[1:],losslist[1:], figure=fig)
        fig2=plt.figure()
        plist=np.reshape(plist, (len(plist), len(plist[0]))).transpose()
        for i in range(len(plist)):
            if self.dfX is None:
                plt.plot(indexlist,plist[i], figure=fig2, label=i)
            else:
                plt.plot(indexlist,plist[i], figure=fig2, label=self.dfX[i][0])
        plt.plot(indexlist ,[0.05] * len(indexlist), '--k', figure=fig2, label="p=0.05")
        plt.legend()
        plt.ylim(0,1)
        fig3=plt.figure()
        mulist=np.reshape(mulist, (len(mulist), len(mulist[0]))).transpose()
        for i in range(len(mulist)):
            if self.dfX is None:
                plt.plot(indexlist,mulist[i], figure=fig3, label=i)
            else:
                plt.plot(indexlist,mulist[i], figure=fig3, label=self.dfX[i][0])
        plt.legend()
        plt.ylim(bottom=0)
        
        self.summary()
    
    def summary(self, **kwargs):
        if self.X is not None or len(self.X)!=0:
            print((self.forward(self.X)["Y"].rsample()-self.Y).mean())
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
                if (prob[i].item()<SNPAutoencoder._pmin): 
                    relevantgene.append(i)
                print(string)
                i+=1
            print("Gene(s) considered relevant:")
            for gene in relevantgene:
                print(self.dfX[gene][0])
        for i in range(len(self.dfX)):
            print("Most important SNP(s) of gene " + self.dfX[i][0] + ":")
            for i2 in range(len(self.list_W_mu[i])):
                if (self.list_W_mu[i][i2] >= 0.1 and self.list_W_alpha[i][i2] <= 1):
                    string=self.dfX[i][1].columns[i2] + ": mu=" + format(self.list_W_mu[i][i2].item(),".3f") + ", alpha=" + format(self.list_W_alpha[i][i2].item(),".3f")
                    print(string)
        
    def genSNPstrand(samplesize, nb_SNP):
        SNP=np.floor(abs(np.random.randn(samplesize, nb_SNP)))
        for sample in SNP:
            for snp in sample:
                if snp>2: snp=2
        return pt.tensor(SNP, device=DEVICE, dtype=float)
    
    @classmethod
    def genSNPprofile(cls,samplesize, nb_gene):
        ret=[]
        for i in range(0,nb_gene):
            nb_SNP=np.random.randint(cls.strand_min,cls.strand_max)
            ret.append(cls.genSNPstrand(samplesize, nb_SNP))
        return ret
    
    @classmethod
    def genfullprofile(cls, samplesize, nb_gene, nb_trait, nb_W2dim, noise = float(0.05)):
        X=cls.genSNPprofile(samplesize, nb_gene)
        W1=[]
        Z=[]
        for i in range(0, nb_gene):
            Wi=abs(np.random.randn(X[i].shape[1]))
            Wi=pt.tensor(Wi, device=DEVICE, dtype=float)
            W1.append(Wi)
            Z.append(X[i] @ Wi)
        Z=pt.stack(Z).transpose(0,1) #Xi * Wi is a vector of shape (1,samplesize) rather than (samplesize,1) so when using pt.stack we have a shape (nbgene, samplesize) so we transpose
        
        W2=pt.zeros(nb_gene, nb_trait, device=DEVICE)
        for i in range(0,nb_W2dim):
            for i2 in range(nb_trait):
                W2[i][i2]=np.random.randn()+1
        Y=Z @ W2.double()
        noise=pt.normal(mean=pt.tensor([[0] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE), 
                    std=pt.tensor([[noise] * nb_trait] * samplesize, dtype=pt.float, device=DEVICE))
        Y=Y + noise
        return ({"X":X, "W1":W1, "Z":Z, "W2":W2, "Y":Y})
    
    def loadGeneticData(linkGenotype = _defaultGL, linkSNPmap= _defaultmap):
        gencsv = pd.read_csv(linkGenotype, header=0, index_col=0)
        
        #we deal with missing values (-1 in the csv)
        gencsv = gencsv[gencsv != -1]
        mean = gencsv.mean(axis=1)
        gencsv=gencsv.transpose() #fillna only works column by colum with series, so we transpose. We would need to transpose later anyway
        gencsv.fillna(mean,inplace=True)
        
        genmap = pd.read_csv(linkSNPmap, header=0, index_col=0)
        return {"genotype":gencsv, "SNP into genes":genmap}
    
    def loadVolumetricData(linkVolume = _defaultVL, linkCognition = _defaultCL):
        volumes = pd.read_csv(linkVolume, header=0, index_col=1).drop("Unnamed: 0", axis=1)
        cognition = pd.read_csv(linkCognition, header=0, index_col=1).drop("Unnamed: 0", axis=1)
        return {"Volume":volumes, "cognition":cognition}
    
    @classmethod
    def loadData(cls, linkGenotype = _defaultGL, linkSNPmap= _defaultmap, linkVolume = _defaultVL, linkCognition = _defaultCL):
        #loading the data
        temp=cls.loadGeneticData(linkGenotype, linkSNPmap)
        genotype=temp["genotype"]
        genmap=temp["SNP into genes"]
        temp=cls.loadVolumetricData(linkVolume, linkCognition)
        volumetric_data=temp["Volume"]
        cognition_data=temp["cognition"]
        physio_dim=-volumetric_data.shape[1]-cognition_data.shape[1]
        
        #we put the data in a better form
        genotype.rename(index = lambda s: int(s[-4:]), inplace=True) #I don't know what the first part of the name is
        #We take the last 4 digits and cast them as int to get the RID in the same type as for the other dataframe
        physio_data=pd.concat([volumetric_data,cognition_data], axis=1, join="inner")
        data=pd.concat([genotype,physio_data], axis=1, join="inner")
        Y=data.iloc[:,physio_dim:]
        #Y=Y[["Hippocampus.bl","CDRSB.bl","ADAS11.bl","MMSE.bl","RAVLT.immediate.bl","RAVLT.forgetting.bl","FAQ.bl"]]
        Y=Y["Hippocampus.bl"]
        tensorY=pt.tensor(Y.values, device=DEVICE, dtype=float)
        X=data.iloc[:,:physio_dim]
        Xlist={}
        for gene in genmap.columns:
            Xi=pd.DataFrame()
            for snp in X.columns:
                if(genmap[gene][snp]==1):
                        Xi=pd.concat([Xi,X[snp]], axis=1)
            Xlist[gene]=Xi
        
        tensorX=[]
        for gene in Xlist:
            tensorX.append(pt.tensor(Xlist[gene].values, device=DEVICE, dtype=float))
        return {"data":data, "X":Xlist, "Tensor X":tensorX, "Y":Y, "Tensor Y":tensorY}