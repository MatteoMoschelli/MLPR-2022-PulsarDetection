# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:46:13 2022

@author: Utente
"""
import utils
import plots
import PCA
import metrics
import hyperparameter_tuning
import GaussianClassifiers
import LogisticRegression
import SupportVectorMachines
import GaussianMixtureModels

print('\t\t\t\t\tPULSAR DETECTION\n')
print('\t\t\t\tTRAINING SET\n')
#-------- Training dataset loading --------#

D_train, L_train = utils.load('data/Train.txt')
print('Dataset loaded\n')

#-------- Observe feature distribution --------#

#plots.plotFeatures(D_train, L_train, utils.featuresNames, utils.classesNames)

#-------- Apply Z-normalization --------#

D_train, _, _ = utils.ZNormalization(D_train)
print('Applied z-normalization')

plots.plotFeatures(D_train, L_train, utils.featuresNames, utils.classesNames)

#-------- Dimensionality analysis (PCA) --------#

PCA.correlation_heatmap(D_train, L_train)

# PCA (m=7)
D_train7 = PCA.compute_PCA(D_train, 7)

# PCA (m=6)
D_train6 = PCA.compute_PCA(D_train, 6)

# PCA (m=5)
D_train5 = PCA.compute_PCA(D_train, 5)

print('Applied PCA with m=7, m=6, m=5\n')

                    #-------- Start training set analysis --------#

            #-------- minDCF --------#

#--- Gaussian Classifiers ---#
"""
print('Starting Gaussian Classifiers analysis:\n')

model_MVG = GaussianClassifiers.GaussianClassifier()
model_diagMVG = GaussianClassifiers.GaussianClassifier_NaiveBayes()
model_tiedMVG = GaussianClassifiers.GaussianClassifier_TiedCovariances()

print('\tSingle Fold approach\n')

print('\t\tZ-norm | no PCA')

(DTR,LTR),(DTE,LTE) = utils.split_db_singleFold(D_train, L_train)

#Full covariance
model_MVG.train(DTR, LTR)
scores = model_MVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tFull covariance --> ', minDCF_05, minDCF_09, minDCF_01)

#Diag covariance
model_diagMVG.train(DTR, LTR)
scores = model_diagMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tDiag covariance --> ', minDCF_05, minDCF_09, minDCF_01)

#Tied covariance
model_tiedMVG.train(DTR, LTR)
scores = model_tiedMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tTied covariance --> ', minDCF_05, minDCF_09, minDCF_01)
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=7)')

(DTR,LTR),(DTE,LTE) = utils.split_db_singleFold(D_train7, L_train)

#Full covariance
model_MVG.train(DTR, LTR)
scores = model_MVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tFull covariance --> ', minDCF_05, minDCF_09, minDCF_01)

#Diag covariance
model_diagMVG.train(DTR, LTR)
scores = model_diagMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tDiag covariance --> ', minDCF_05, minDCF_09, minDCF_01)

#Tied covariance
model_tiedMVG.train(DTR, LTR)
scores = model_tiedMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tTied covariance --> ', minDCF_05, minDCF_09, minDCF_01)
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=6)')

(DTR,LTR),(DTE,LTE) = utils.split_db_singleFold(D_train6, L_train)

#Full covariance
model_MVG.train(DTR, LTR)
scores = model_MVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tFull covariance --> ', minDCF_05, minDCF_09, minDCF_01)

#Diag covariance
model_diagMVG.train(DTR, LTR)
scores = model_diagMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tDiag covariance --> ', minDCF_05, minDCF_09, minDCF_01)

#Tied covariance
model_tiedMVG.train(DTR, LTR)
scores = model_tiedMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tTied covariance --> ', minDCF_05, minDCF_09, minDCF_01)
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=5)')

(DTR,LTR),(DTE,LTE) = utils.split_db_singleFold(D_train5, L_train)

#Full covariance
model_MVG.train(DTR, LTR)
scores = model_MVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tFull covariance --> ', minDCF_05, minDCF_09, minDCF_01)

#Diag covariance
model_diagMVG.train(DTR, LTR)
scores = model_diagMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tDiag covariance --> ', minDCF_05, minDCF_09, minDCF_01)

#Tied covariance
model_tiedMVG.train(DTR, LTR)
scores = model_tiedMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tTied covariance --> ', minDCF_05, minDCF_09, minDCF_01)


print('\tK-Fold approach\n')

print('\t\tZ-norm | no PCA')

print('\t\t\tFull covariance --> ', utils.KFold(D_train, L_train, GaussianClassifiers.GaussianClassifier()))
print('\t\t\tDiag covariance --> ', utils.KFold(D_train, L_train, GaussianClassifiers.GaussianClassifier_NaiveBayes()))
print('\t\t\tTied covariance --> ', utils.KFold(D_train, L_train, GaussianClassifiers.GaussianClassifier_TiedCovariances()))
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=7)')

print('\t\t\tFull covariance --> ', utils.KFold(D_train7, L_train, GaussianClassifiers.GaussianClassifier()))
print('\t\t\tDiag covariance --> ', utils.KFold(D_train7, L_train, GaussianClassifiers.GaussianClassifier_NaiveBayes()))
print('\t\t\tTied covariance --> ', utils.KFold(D_train7, L_train, GaussianClassifiers.GaussianClassifier_TiedCovariances()))
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=6)')

print('\t\t\tFull covariance --> ', utils.KFold(D_train6, L_train, GaussianClassifiers.GaussianClassifier()))
print('\t\t\tDiag covariance --> ', utils.KFold(D_train6, L_train, GaussianClassifiers.GaussianClassifier_NaiveBayes()))
print('\t\t\tTied covariance --> ', utils.KFold(D_train6, L_train, GaussianClassifiers.GaussianClassifier_TiedCovariances()))
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=5)')

print('\t\t\tFull covariance --> ', utils.KFold(D_train5, L_train, GaussianClassifiers.GaussianClassifier()))
print('\t\t\tDiag covariance --> ', utils.KFold(D_train5, L_train, GaussianClassifiers.GaussianClassifier_NaiveBayes()))
print('\t\t\tTied covariance --> ', utils.KFold(D_train5, L_train, GaussianClassifiers.GaussianClassifier_TiedCovariances()))

#--- Logistic Regression ---#

print('Starting Logistic Regression analysis:\n')

# linear LR tuning
hyperparameter_tuning.linear_LR_tuning(D_train, L_train, mode = 'singleFold')
hyperparameter_tuning.linear_LR_tuning(D_train, L_train, mode = 'KFold')
hyperparameter_tuning.linear_LR_tuning(D_train7, L_train, mode = 'singleFold')
hyperparameter_tuning.linear_LR_tuning(D_train7, L_train, mode = 'KFold')
"""
linear_LR_lambda = 1e-5
model_linearLR = LogisticRegression.LinearLR()

print('Selected value for lambda: ', linear_LR_lambda)

print('\tK-Fold approach\n')
"""
print('\t\tZ-norm | no PCA')
print(utils.KFoldLR(D_train, L_train, model_linearLR, linear_LR_lambda))
#---------------------------------------------------------------------------------------------------------

print('\t\tZ-norm | PCA(m=7)')
print(utils.KFoldLR(D_train7, L_train, model_linearLR, linear_LR_lambda))
"""
# print('A')
# hyperparameter_tuning.quadratic_LR_tuning(D_train, L_train, mode='singleFold')
# print('B')
# hyperparameter_tuning.quadratic_LR_tuning(D_train, L_train, mode='KFold')
# print('C')
# hyperparameter_tuning.quadratic_LR_tuning(D_train7, L_train, mode='singleFold')
# print('D')
# hyperparameter_tuning.quadratic_LR_tuning(D_train7, L_train, mode='KFold')
# print('E')

# hyperparameter_tuning.balanced_linear_SVM_tuning(D_train, L_train, mode='KFold')
# print('F')
# hyperparameter_tuning.balanced_linear_SVM_tuning(D_train7, L_train, mode='KFold')
# print('G')



#### SVM (linear + polynomial & RBF kernel)
#### GMM
print('A')
hyperparameter_tuning.GMM_tuning(D_train, L_train, mode='KFold')
print('B')
hyperparameter_tuning.GMM_tuning(D_train7, L_train, mode='KFold')
print('C')
hyperparameter_tuning.diag_GMM_tuning(D_train, L_train, mode='KFold')
print('D')
hyperparameter_tuning.diag_GMM_tuning(D_train7, L_train, mode='KFold')
print('E')
hyperparameter_tuning.tied_GMM_tuning(D_train, L_train, mode='KFold')
print('F')
hyperparameter_tuning.tied_GMM_tuning(D_train7, L_train, mode='KFold')
## actDCF
#### MVG
#### LR (linear + quad)
#### SVM (linear + polynomial & RBF kernel)
#### GMM
## score calibration for best models
# test set analysis
## minDCF
#### MVG
#### LR (linear + quad)
#### SVM (linear + polynomial & RBF kernel)
#### GMM
## actDCF
#### MVG
#### LR (linear + quad)
#### SVM (linear + polynomial & RBF kernel)
#### GMM
## score calibration for best models
