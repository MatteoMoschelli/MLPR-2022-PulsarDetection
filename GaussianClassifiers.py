# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:09:58 2022

@author: Matteo
"""

import utils
import numpy as np
import scipy

def logpdf_GAU_ND(x, mu, C):
    return -(x.shape[0]/2)*np.log(2*np.pi)-(0.5)*(np.linalg.slogdet(C)[1])- (0.5)*np.multiply((np.dot((x-mu).T, np.linalg.inv(C))).T,(x-mu)).sum(axis=0)

class GaussianClassifier:
    
    def train(self, D, L):
        DTR0 = D[:, L==0] #training samples of class 0
        DTR1 = D[:, L==1] #training samples of class 1
        
        self.mean0 = utils.vcol(DTR0.mean(axis=1))
        self.mean1 = utils.vcol(DTR1.mean(axis=1))
       
        self.cov0 = np.cov(D[:, L == 0])
        self.cov1 = np.cov(D[:, L == 1])
        
    def compute_scores(self, X):
        like0 = logpdf_GAU_ND(X, self.mean0, self.cov0 )
        like1 = logpdf_GAU_ND(X, self.mean1, self.cov1 )
        
        #log-likelihood ratios
        llr = like1-like0
        return llr
    

class GaussianClassifier_NaiveBayes:
    
    def train(self, D, L):
        DTR0 = D[:, L==0] #training samples of class 0
        DTR1 = D[:, L==1] #training samples of class 1
        
        self.mean0 = utils.vcol(DTR0.mean(axis=1))
        self.mean1 = utils.vcol(DTR1.mean(axis=1))
       
        self.cov0 = np.multiply(np.cov(D[:, L == 0]), np.eye(DTR0.shape[0]))
        self.cov1 = np.multiply(np.cov(D[:, L == 1]), np.eye(DTR1.shape[0]))
    
    def compute_scores(self, X):
        like0 = logpdf_GAU_ND(X, self.mean0, self.cov0 )
        like1 = logpdf_GAU_ND(X, self.mean1, self.cov1 )
        
        #log-likelihood ratios
        llr = like1-like0
        return llr
    
class GaussianClassifier_TiedCovariances:
    
    def train(self, D, L):
        DTR0 = D[:, L==0] #training samples of class 0
        DTR1 = D[:, L==1] #training samples of class 1
        
        self.mean0 = utils.vcol(DTR0.mean(axis=1))
        self.mean1 = utils.vcol(DTR1.mean(axis=1))
         
        self.cov0 = np.cov(DTR0)
        self.cov1 = np.cov(DTR1)

        self.cov = 1/(D.shape[1]) * (DTR0.shape[1] * self.cov0 + DTR1.shape[1] * self.cov1)         
        
    def compute_scores(self, X):
        like0 = logpdf_GAU_ND(X, self.mean0, self.cov )
        like1 = logpdf_GAU_ND(X, self.mean1, self.cov )
        
        #log-likelihood ratios
        llr = like1-like0
        return llr