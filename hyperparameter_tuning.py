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

priors = [0.5, 0.9, 0.1]

def linear_LR_tuning(D, L, mode = 'singleFold'):
    '''
    values for "mode": 
        singleFold: compute minDCF with single fold\n
        KFold: compute minDCF with KFold cross-validation
    '''
        
    model = LogisticRegression.LogisticRegression()
    lambdas = np.logspace(-5, 5, 50)
    
    if(mode == 'KFold'):
        minDCF = [([utils.KFoldLR(D, L, model, l, K = 5, prior = p) for p in priors], l) for l in tqdm(lambdas)]
    elif(mode == 'singleFold'):
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
