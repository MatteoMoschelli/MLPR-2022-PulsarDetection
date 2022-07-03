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

#hyperparameter_tuning.balanced_linear_SVM_tuning(D, L, mode='KFold')

GMM1 = GaussianMixtureModels.GMM()
GMM1.train(D,L,M=4)
s = GMM1.predictAndGetScores(DTE)
print(utils.metrics.compute_minDCF(s,LTE,0.5,1,1))