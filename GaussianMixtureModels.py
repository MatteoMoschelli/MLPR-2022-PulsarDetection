# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:28:34 2022

@author: Matteo
"""

import numpy as np
import utils
import scipy.special 
import LBG_Algorithm

def logpdf_GAU_ND(x, mu, C):
    return -(x.shape[0]/2)*np.log(2*np.pi)-(0.5)*(np.linalg.slogdet(C)[1])- (0.5)*np.multiply((np.dot((x-mu).T, np.linalg.inv(C))).T,(x-mu)).sum(axis=0)

def GMM_loglikelihood(X, gmm):
    tempSum=np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        tempSum[i,:]=np.log(gmm[i][0]) + logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return scipy.special.logsumexp(tempSum, axis=0)

class GMM():
    
    def train(self, D, L, M):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]
        
        GMM0_init = [(1.0, utils.vcol(D0.mean(axis=1)), np.cov(D0))]
        GMM1_init = [(1.0, utils.vcol(D1.mean(axis=1)), np.cov(D1))]
        
        self.GMM0 = LBG_Algorithm.LBG(GMM0_init, D0, M)
        self.GMM1 = LBG_Algorithm.LBG(GMM1_init, D1, M)
        
    def predictAndGetScores(self, X):
        
        LS0 = GMM_loglikelihood(X, self.GMM0)
        LS1 = GMM_loglikelihood(X, self.GMM1)
        
        llr = LS1-LS0
        return llr
    
    
class GMMDiag():
    
    def train(self, D, L, M):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]
        
        GMM0_init = [(1.0, utils.vcol(D0.mean(axis=1)), np.cov(D0) * np.eye(D0.shape[0]))]
        GMM1_init = [(1.0, utils.vcol(D1.mean(axis=1)), np.cov(D1) * np.eye(D1.shape[0]))]
        
        self.GMM0 = LBG_Algorithm.DiagLBG(GMM0_init, D0, M)
        self.GMM1 = LBG_Algorithm.DiagLBG(GMM1_init, D1, M)
        
    def predictAndGetScores(self, X):
        
        LS0 = GMM_loglikelihood(X, self.GMM0)
        LS1 = GMM_loglikelihood(X, self.GMM1)
        
        llr = LS1-LS0
        return llr
    

class GMMTiedCov():
    
    def train(self, D, L, M):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]
        
        C0 = np.cov(D0)
        C1 = np.cov(D1)
        self.C = 1/(D.shape[1]) * (D0.shape[1] * C0 + D1.shape[1] * C1)
        
        GMM0_init = [(1.0, utils.vcol(D0.mean(axis=1)), self.C)]
        GMM1_init = [(1.0, utils.vcol(D1.mean(axis=1)), self.C)]
        
        self.GMM0 = LBG_Algorithm.TiedLBG(GMM0_init, D0, M)
        self.GMM1 = LBG_Algorithm.TiedLBG(GMM1_init, D1, M)
        
    def predictAndGetScores(self, X):
        LS0 = GMM_loglikelihood(X, self.GMM0)
        LS1 = GMM_loglikelihood(X, self.GMM1)
        
        llr = LS1-LS0
        return llr