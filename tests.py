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
import matplotlib.pyplot as plt

D_train, L_train = utils.load('data/Train.txt')
D_train, _, _ = utils.ZNormalization(D_train)
D_train7 = PCA.compute_PCA(D_train, 7)


# print('\tK-Fold approach\n')
# print('\t\tZ-norm | no PCA')

# model = SupportVectorMachines.BalancedLinearSVM()

# print('Linear SVM (pi_t=0.9) --> ', utils.KFoldSVM(D_train, L_train, model, C=1e-3, K=3, prior=0.5, pi_T=0.9))
# print('Linear SVM (pi_t=0.5) --> ', utils.KFoldSVM(D_train, L_train, model, C=1e-3, K=3, prior=0.5, pi_T=0.5))
# print('Linear SVM (pi_t=0.1) --> ', utils.KFoldSVM(D_train, L_train, model, C=1e-3, K=3, prior=0.5, pi_T=0.1))

# print('\t\tZ-norm | PCA(m=7)')

# model = SupportVectorMachines.BalancedLinearSVM()

# print('Linear SVM (pi_t=0.9) --> ', utils.KFoldSVM(D_train7, L_train, model, C=1e-3, K=3, prior=0.5, pi_T=0.9))
# print('Linear SVM (pi_t=0.5) --> ', utils.KFoldSVM(D_train7, L_train, model, C=1e-3, K=3, prior=0.5, pi_T=0.5))
# print('Linear SVM (pi_t=0.1) --> ', utils.KFoldSVM(D_train7, L_train, model, C=1e-3, K=3, prior=0.5, pi_T=0.1))


M = 8
model_full = GaussianMixtureModels.GMM()
model_diag = GaussianMixtureModels.GMMDiag()
model_tied = GaussianMixtureModels.GMMTiedCov()

print('\tK-Fold approach\n')
print('\t\tZ-norm | no PCA')

print('Full-Cov GMM --> ', utils.KFoldGMM(D_train, L_train, model_full, K=3, M=M))
print('Diag-Cov GMM --> ', utils.KFoldGMM(D_train, L_train, model_diag, K=3, M=M))
print('Tied-Cov GMM --> ', utils.KFoldGMM(D_train, L_train, model_tied, K=3, M=M))

print('\t\tZ-norm | PCA(m=7)')

print('Full-Cov GMM --> ', utils.KFoldGMM(D_train7, L_train, model_full, K=3, M=M))
print('Diag-Cov GMM --> ', utils.KFoldGMM(D_train7, L_train, model_diag, K=3, M=M))
print('Tied-Cov GMM --> ', utils.KFoldGMM(D_train7, L_train, model_tied, K=3, M=M))




















# D, L = utils.load('../WineQualityDetection/Train.txt')
# D_gauss = utils.Gaussianization(D,D)


# D = PCA.compute_PCA(D_gauss,10)

# (DTR,LTR),(DTE,LTE) = utils.split_db_singleFold(D,L)

# #hyperparameter_tuning.linear_LR_tuning(D, L, mode = 'singleFold')
# #hyperparameter_tuning.linear_LR_tuning(D, L, mode = 'KFold')


# # SVM.train(DTR,LTR, kernel='RBF')
# # s = SVM.predictAndGetScores(DTE)
# # print(utils.metrics.compute_minDCF(s,LTE,0.5,1,1))

# #hyperparameter_tuning.balanced_RBF_SVM_tuning(D, L, mode='KFold')
# #hyperparameter_tuning.balanced_RBF_SVM_tuning(D, L, mode='singleFold')

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
# M_params = [2,4,8,16,32,64]
# plots.plotDCF_GMM(M_params, DCFs_gauss, "M", "min DCF")

# #hyperparameter_tuning.balanced_RBF_SVM_tuning(D, L, mode='singleFold')