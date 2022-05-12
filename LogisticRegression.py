# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:46:04 2022

@author: Matteo
"""

import numpy as np
import scipy.optimize
import utils

class LogisticRegression:
    
    def train(self, D, L, l, prior=0.5):    # l = lambda
        self.x, self.f, self.d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0 = np.zeros(D.shape[0] + 1), args=(D, L, l, prior), approx_grad=True)
        
    def predictAndGetScores(self, X):
        scores = np.dot(self.x[0:-1], X) + self.x[-1]
        return scores
    
    def logreg_obj(self, v, D, L, l, prior):
        w, b = v[0:-1], v[-1]
        return l/2 * scipy.linalg.norm(w) ** 2 + prior/D[:, L==1].shape[1] * np.sum(L * np.log1p(np.exp(-np.dot(w.T, D) - b))) + (1-prior)/D[:, L==0].shape[1] * np.sum((1 - L) * np.log1p(np.exp(np.dot(w.T, D) + b)))
