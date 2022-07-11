# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:48:11 2022

@author: Matteo
"""

import EM_Algorithm
import numpy as np

def split(GMM, alpha = 0.1):
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        U, s, Vh = np.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    return splittedGMM

def LBG(GMM, X, iterations):
    while True:
        GMM = split(GMM)
        #print("\t\tM = " + str(len(GMM)) + "/"+str(iterations))
        GMM = EM_Algorithm.EMalgorithm(X, GMM)
        
        if(len(GMM) >= iterations):
            break
    return GMM

def DiagLBG(GMM, X, iterations):
    while True:
        GMM = split(GMM)
        #print("\t\tM = " + str(len(GMM)) + "/"+str(iterations))
        GMM = EM_Algorithm.DiagEMalgorithm(X, GMM)
        
        if(len(GMM) >= iterations):
            break
    return GMM


def TiedLBG(GMM, X, iterations):    
    while True:
        GMM = split(GMM)
        #print("\t\tM = " + str(len(GMM)) + "/"+str(iterations))
        GMM = EM_Algorithm.TiedEMalgorithm(X, GMM)
        
        if(len(GMM) >= iterations):
            break
    return GMM