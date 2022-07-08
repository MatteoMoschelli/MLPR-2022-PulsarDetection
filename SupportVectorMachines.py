# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:05:01 2022

@author: Matteo
"""

import numpy as np
import scipy.optimize
import utils
from itertools import repeat


def SVM_obj(alpha, H):
    grad = np.dot(H, alpha) - np.ones(H.shape[1])
    return ((1/2) * np.dot(np.dot(alpha.T, H), alpha) - np.dot(alpha.T, np.ones(H.shape[1])), grad)

def SVM_dual_formulation(DTR, LTR, C, K, piT):
    '''
    values for "mode": 
        balanced: apply class re-balancing\n
        unbalanced: apply the default SVM formulation
    '''
    
    # Compute the D matrix for the extended training set
    row = np.zeros(DTR.shape[1]) + K
    D = np.vstack((DTR, row))

    # Compute the H matrix 
    Gij = np.dot(D.T, D)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    
    # Prepare box constraints
    DTR_0 = DTR[:,LTR == 0]
    DTR_1 = DTR[:,LTR == 1]
    pi_f_emp = DTR_0.shape[1] / DTR.shape[1]
    pi_t_emp = DTR_1.shape[1] / DTR.shape[1]
    
    C0 = (C * (1 - piT)) / pi_f_emp
    C1 = (C * piT) / pi_t_emp
    
    b = [(0, C1) if LTR[i] == 1 else (0, C0) for i in range(DTR.shape[1])]
    
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(SVM_obj, np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=0, factr=1.0)
    w = np.sum((x * LTR).reshape(1, DTR.shape[1]) * D, axis=1)
    
    return w
        
def SVM_dual_formulation_kernel(DTR, LTR, C, K, kernel='poly', piT=0.5, c=0, d=2, gamma=1.0):
    if kernel == 'poly':
        kernelFunction = poly_kernel(DTR, K, c, d)
    elif kernel == 'RBF':
        kernelFunction = RBF_kernel(DTR, DTR, K, gamma)
    
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    
    # Prepare box constraints
    DTR_0 = DTR[:,LTR == 0]
    DTR_1 = DTR[:,LTR == 1]
    pi_f_emp = DTR_0.shape[1] / DTR.shape[1]
    pi_t_emp = DTR_1.shape[1] / DTR.shape[1]
    
    C0 = (C * (1 - piT)) / pi_f_emp
    C1 = (C * piT) / pi_t_emp
        
    b = [(0, C1) if LTR[i] == 1 else (0, C0) for i in range(DTR.shape[1])]
    
    
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(SVM_obj, np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=0, factr=1.0)
    return x

def poly_kernel(D, K, c, d):
    return (np.dot(D.T, D) + c)**d + K**2

def RBF_kernel(X, Y, K, gamma):
    result = np.zeros((X.shape[1], Y.shape[1]))
    
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            result[i][j] = np.exp(-gamma * (np.linalg.norm(X[:, i] - Y[:, j])**2)) + K**2
            
    return result

class BalancedLinearSVM:
    
    def train(self, D, L, C=1.0, K=1.0, p_t=0.5):
        self.DTR = D
        self.LTR = L
        self.K = K
        self.C = C 
        self.p_t = p_t
        
        self.w = SVM_dual_formulation(self.DTR, self.LTR, self.C, self.K, self.p_t)

    def predictAndGetScores(self, X):
        DTE = np.vstack([X, np.zeros(X.shape[1]) + self.K])
        S = np.dot(self.w.T, DTE)
        return S
    

class BalancedQuadraticSVM:
    
    def train(self, D, L, kernel='poly', C=1.0, K=1.0, p_t=0.5, c=0, d=2, gamma=1.0):
        self.kernel = kernel
        self.DTR = D
        self.LTR = L
        self.K = K
        self.C = C 
        self.p_t = p_t
        
        if kernel == 'poly':
            self.c = c
            self.d = d
            
            self.x = SVM_dual_formulation_kernel(self.DTR, self.LTR, self.C, self.K, kernel=self.kernel, piT=self.p_t, c=self.c, d=self.d)
            
        elif kernel == 'RBF':
            self.gamma = gamma
            
            self.x = SVM_dual_formulation_kernel(self.DTR, self.LTR, self.C, self.K, kernel=self.kernel, piT=self.p_t, gamma=self.gamma)
            
    def predictAndGetScores(self, X):
        if self.kernel == 'poly':
            S = np.sum(np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (np.dot(self.DTR.T, X) + self.c)**self.d + self.K**2), axis=0)
            return S
        elif self.kernel == 'RBF':
            kernelFunction = RBF_kernel(self.DTR, X, self.K, self.gamma)
            S = np.sum(np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), kernelFunction), axis=0)
            return S
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            