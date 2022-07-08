# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:44:01 2022

@author: Matteo
"""

from tqdm import tqdm
import numpy as np

import utils
import metrics
import plots
import LogisticRegression
import SupportVectorMachines
import GaussianMixtureModels

priors = [0.5, 0.9, 0.1]

def linear_LR_tuning(D, L, mode = 'singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
        
    model = LogisticRegression.LinearLR()
    lambdas = np.logspace(-5, 2, 30)
    
    if mode == 'KFold':
        minDCF = [([utils.KFoldLR(D, L, model, l, K = 3, prior = p) for p in priors], l) for l in tqdm(lambdas)]
    elif mode == 'singleFold' :
        (DTR, LTR),(DTE, LTE) = utils.split_db_singleFold(D, L)
        minDCF = []
        
        for l in tqdm(lambdas):
            model.train(DTR, LTR, l, prior = 0.5)
            scores = model.predictAndGetScores(DTE)
            
            minDCF.append(([metrics.compute_minDCF(scores, LTE, p, 1, 1) for p in priors], l))
    else:
        print("Invalid parameter: mode")
        return
    
    results = min(minDCF, key=lambda t: t[0][0])
    
    print(f"Result on minimum DCF over 0.5 application:\n\
          selected lambda: {results[1]}\n\
          prior = 0.5, minDCF {results[0][0]}\n\
          prior = 0.1, minDCF {results[0][1]}\n\
          prior = 0.9, minDCF {results[0][2]}\n")
    
    y0 = [minDCF[i][0][0] for i in range(len(minDCF))]
    y1 = [minDCF[i][0][1] for i in range(len(minDCF))]
    y2 = [minDCF[i][0][2] for i in range(len(minDCF))]
    
    plots.plotDCF(lambdas, (y0, y1, y2), "λ", "min DCF")
  
    return results[1]


def quadratic_LR_tuning(D, L, mode = 'singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
    
    model = LogisticRegression.QuadraticLR()
    lambdas = np.logspace(-5, 2, 30)
    
    if mode == 'KFold':
        minDCF = [([utils.KFoldLR(D, L, model, l, K = 3, prior = p) for p in priors], l) for l in tqdm(lambdas)]
    elif mode == 'singleFold':
        (DTR, LTR),(DTE, LTE) = utils.split_db_singleFold(D, L)
        minDCF = []
        
        for l in tqdm(lambdas):
            model.train(DTR, LTR, l, prior = 0.5)
            scores = model.predictAndGetScores(DTE)
            
            minDCF.append(([metrics.compute_minDCF(scores, LTE, p, 1, 1) for p in priors], l))
    else:
        print("Invalid parameter: mode")
        return
    
    results = min(minDCF, key=lambda t: t[0][0])
    
    print(f"Result on minimum DCF over 0.5 application:\n\
          selected lambda: {results[1]}\n\
          prior = 0.5, minDCF {results[0][0]}\n\
          prior = 0.1, minDCF {results[0][1]}\n\
          prior = 0.9, minDCF {results[0][2]}\n")
    
    y0 = [minDCF[i][0][0] for i in range(len(minDCF))]
    y1 = [minDCF[i][0][1] for i in range(len(minDCF))]
    y2 = [minDCF[i][0][2] for i in range(len(minDCF))]
    
    plots.plotDCF(lambdas, (y0, y1, y2), "λ", "min DCF")
  
    return results[1]

def balanced_linear_SVM_tuning(D, L, mode='singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
    
    model = SupportVectorMachines.BalancedLinearSVM()
    C_params = np.logspace(-5, -2, 30)
    
    if mode == 'KFold':
        minDCF = [([utils.KFoldSVM(D, L, model, C, K = 3, prior = p) for p in priors], C) for C in tqdm(C_params)]
    elif mode == 'singleFold':
        (DTR, LTR),(DTE, LTE) = utils.split_db_singleFold(D, L)
        minDCF = []
        
        for C in tqdm(C_params):
            model.train(DTR, LTR, C)
            scores = model.predictAndGetScores(DTE)
            
            minDCF.append(([metrics.compute_minDCF(scores, LTE, p, 1, 1) for p in priors], C))
    else:
        print("Invalid parameter: mode")
        return
    
    results = min(minDCF, key=lambda t: t[0][0])
    
    print(f"Result on minimum DCF over 0.5 application:\n\
          selected C: {results[1]}\n\
          prior = 0.5, minDCF {results[0][0]}\n\
          prior = 0.1, minDCF {results[0][1]}\n\
          prior = 0.9, minDCF {results[0][2]}\n")
    
    y0 = [minDCF[i][0][0] for i in range(len(minDCF))]
    y1 = [minDCF[i][0][1] for i in range(len(minDCF))]
    y2 = [minDCF[i][0][2] for i in range(len(minDCF))]
    
    plots.plotDCF(C_params, (y0, y1, y2), "C", "min DCF")
    
    return results[1]

def balanced_poly_SVM_tuning(D, L, mode='singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
    
    model = SupportVectorMachines.BalancedQuadraticSVM()
    C_params = np.logspace(-5, -2, 20)
    c_params = [0, 1, 10]
    
    if mode == 'KFold':
        minDCF = np.array([([utils.KFoldSVM_kernel(D, L, model, kernel='poly', C=C, K=3, prior=0.5, c=c_i) for c_i in c_params], C) for C in tqdm(C_params)], dtype=object)

    
        y0 = [minDCF[i][0][0] for i in range(len(C_params))]
        y1 = [minDCF[i][0][1] for i in range(len(C_params))]
        y2 = [minDCF[i][0][2] for i in range(len(C_params))]
        #y3 = [minDCF[i][0][3] for i in range(len(C_params))]
        
        plots.plotDCF_poly(C_params, (y0, y1, y2), "C", "min DCF")
        
    elif mode == 'singleFold':
        (DTR, LTR),(DTE, LTE) = utils.split_db_singleFold(D, L)
        minDCF = []
        
        for c in c_params:
            for C in tqdm(C_params):
                model.train(DTR, LTR, kernel='poly', C=C, c=c)
                scores = model.predictAndGetScores(DTE)
                
                minDCF.append(metrics.compute_minDCF(scores, LTE, 0.5, 1, 1))
                
        y0 = minDCF[0:len(C_params)]
        y1 = minDCF[len(C_params):2*len(C_params)]
        y2 = minDCF[2*len(C_params):3*len(C_params)]
        
        plots.plotDCF_poly(C_params, (y0, y1, y2), "C", "min DCF")
    else:
        print("Invalid parameter: mode")
        return


def balanced_RBF_SVM_tuning(D, L, mode='singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
    
    model = SupportVectorMachines.BalancedQuadraticSVM()
    C_params = np.logspace(-5, -2, 20)
    gamma_params = [1e-5, 1e-4, 1e-3]
    
    if mode == 'KFold':
        minDCF = np.array([([utils.KFoldSVM_kernel(D, L, model, kernel='RBF', C=C, K=3, prior=0.5, gamma=gamma_i) for gamma_i in gamma_params], C) for C in tqdm(C_params)], dtype=object)

    
        y0 = [minDCF[i][0][0] for i in range(len(C_params))]
        y1 = [minDCF[i][0][1] for i in range(len(C_params))]
        y2 = [minDCF[i][0][2] for i in range(len(C_params))]
        #y3 = [minDCF[i][0][3] for i in range(len(C_params))]
        
        plots.plotDCF_RBF(C_params, (y0, y1, y2), "C", "min DCF")
        
    elif mode == 'singleFold':
        (DTR, LTR),(DTE, LTE) = utils.split_db_singleFold(D, L)
        minDCF = []
        
        for gamma in gamma_params:
            for C in tqdm(C_params):
                model.train(DTR, LTR, kernel='RBF', C=C, gamma=gamma)
                scores = model.predictAndGetScores(DTE)
                
                minDCF.append(metrics.compute_minDCF(scores, LTE, 0.5, 1, 1))
                
        y0 = minDCF[0:len(C_params)]
        y1 = minDCF[len(C_params):2*len(C_params)]
        y2 = minDCF[2*len(C_params):3*len(C_params)]
        
        plots.plotDCF_RBF(C_params, (y0, y1, y2), "C", "min DCF")
    else:
        print("Invalid parameter: mode")
        return
    
def GMM_tuning(D, L, mode='singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
    
    model = GaussianMixtureModels.GMM()
    M_params = [2,4,8,16,32,64]

    if mode == 'KFold':
        minDCF_05 = []
        minDCF_09 = []
        minDCF_01 = []
        
        for M in tqdm(M_params):
            minDCFs = np.array(utils.KFoldGMM(D, L, model, 3, M))
            minDCF_05.append(minDCFs[0])
            minDCF_09.append(minDCFs[1])
            minDCF_01.append(minDCFs[2])
        
    elif mode == 'singleFold':
        (DTR, LTR),(DTE, LTE) = utils.split_db_singleFold(D, L)
        minDCF_05 = []
        minDCF_09 = []
        minDCF_01 = []
        
        for M in tqdm(M_params):
            model.train(DTR, LTR, M)
            scores = model.predictAndGetScores(DTE)
            
            minDCF_05.append(metrics.compute_minDCF(scores, LTE, 0.5, 1, 1))
            minDCF_09.append(metrics.compute_minDCF(scores, LTE, 0.9, 1, 1))
            minDCF_01.append(metrics.compute_minDCF(scores, LTE, 0.1, 1, 1))
    else:
        print("Invalid parameter: mode")
        return
    
    
    minDCF_05 = np.array(minDCF_05)
    minDCF_09 = np.array(minDCF_09)
    minDCF_01 = np.array(minDCF_01)
    
    plots.plotDCF_GMM(M_params, minDCF_05,minDCF_09, minDCF_01, "M", "min DCF")
    
def diag_GMM_tuning(D, L, mode='singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
    
    model = GaussianMixtureModels.GMMDiag()
    M_params = [2,4,8,16,32,64]
    
    
    if mode == 'KFold':
        minDCF_05 = []
        minDCF_09 = []
        minDCF_01 = []
        
        for M in tqdm(M_params):
            minDCFs = np.array(utils.KFoldGMM(D, L, model, 3, M))
            minDCF_05.append(minDCFs[0])
            minDCF_09.append(minDCFs[1])
            minDCF_01.append(minDCFs[2])
        
    elif mode == 'singleFold':
        (DTR, LTR),(DTE, LTE) = utils.split_db_singleFold(D, L)
        minDCF_05 = []
        minDCF_09 = []
        minDCF_01 = []
        
        for M in tqdm(M_params):
            model.train(DTR, LTR, M)
            scores = model.predictAndGetScores(DTE)
            
            minDCF_05.append(metrics.compute_minDCF(scores, LTE, 0.5, 1, 1))
            minDCF_09.append(metrics.compute_minDCF(scores, LTE, 0.9, 1, 1))
            minDCF_01.append(metrics.compute_minDCF(scores, LTE, 0.1, 1, 1))
    else:
        print("Invalid parameter: mode")
        return
    
    
    minDCF_05 = np.array(minDCF_05)
    minDCF_09 = np.array(minDCF_09)
    minDCF_01 = np.array(minDCF_01)
    
    plots.plotDCF_GMM(M_params, minDCF_05,minDCF_09, minDCF_01, "M", "min DCF")
    
def tied_GMM_tuning(D, L, mode='singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
    
    model = GaussianMixtureModels.GMMTiedCov()
    M_params = [2,4,8,16,32,64]
    
    if mode == 'KFold':
        minDCF_05 = []
        minDCF_09 = []
        minDCF_01 = []
        
        for M in tqdm(M_params):
            minDCFs = np.array(utils.KFoldGMM(D, L, model, 3, M))
            minDCF_05.append(minDCFs[0])
            minDCF_09.append(minDCFs[1])
            minDCF_01.append(minDCFs[2])
        
    elif mode == 'singleFold':
        (DTR, LTR),(DTE, LTE) = utils.split_db_singleFold(D, L)
        minDCF_05 = []
        minDCF_09 = []
        minDCF_01 = []
        
        for M in tqdm(M_params):
            model.train(DTR, LTR, M)
            scores = model.predictAndGetScores(DTE)
            
            minDCF_05.append(metrics.compute_minDCF(scores, LTE, 0.5, 1, 1))
            minDCF_09.append(metrics.compute_minDCF(scores, LTE, 0.9, 1, 1))
            minDCF_01.append(metrics.compute_minDCF(scores, LTE, 0.1, 1, 1))
    else:
        print("Invalid parameter: mode")
        return
    
    
    minDCF_05 = np.array(minDCF_05)
    minDCF_09 = np.array(minDCF_09)
    minDCF_01 = np.array(minDCF_01)
    
    plots.plotDCF_GMM(M_params, minDCF_05,minDCF_09, minDCF_01, "M", "min DCF")
    