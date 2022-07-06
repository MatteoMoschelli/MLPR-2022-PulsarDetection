# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:42:49 2022

@author: Matteo
"""

import utils
import plots
import PCA
import metrics
import hyperparameter_tuning
import LogisticRegression
import SupportVectorMachines
import GaussianMixtureModels
import numpy as np
import plots


D, L = utils.load('../WineQualityDetection/Train.txt')
D_gauss = utils.Gaussianization(D,D)


D = PCA.compute_PCA(D_gauss,10)

(DTR,LTR),(DTE,LTE) = utils.split_db_singleFold(D,L)

#hyperparameter_tuning.linear_LR_tuning(D, L, mode = 'singleFold')
#hyperparameter_tuning.linear_LR_tuning(D, L, mode = 'KFold')


# SVM.train(DTR,LTR, kernel='RBF')
# s = SVM.predictAndGetScores(DTE)
# print(utils.metrics.compute_minDCF(s,LTE,0.5,1,1))

#hyperparameter_tuning.balanced_RBF_SVM_tuning(D, L, mode='KFold')
#hyperparameter_tuning.balanced_RBF_SVM_tuning(D, L, mode='singleFold')

# def load_GMM_DCFs(filename_gauss):
#     file = open(filename_gauss,'r')
    
#     DCFs_list = []
#     for line in file:
#         fields = line.split(',')
#         DCFs_list.append(float(fields[1]))
#     file.close()
    
#     DCFs_gauss = np.array(DCFs_list)
    
    
    
#     return DCFs_gauss

# DCFs_gauss = load_GMM_DCFs('GMM_dcf_gauss.txt')
# M_params = [1,2,4,8,16,32,62,128,256,512]
# plots.plotDCF_GMM(M_params, DCFs_gauss, "M", "min DCF")

hyperparameter_tuning.balanced_RBF_SVM_tuning(D, L, mode='singleFold')