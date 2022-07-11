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
print('Training Dataset loaded\n')

#-------- Observe feature distribution --------#

#plots.plotFeatures(D_train, L_train, utils.featuresNames, utils.classesNames)

#-------- Apply Z-normalization --------#

D_train, _, _ = utils.ZNormalization(D_train)
print('Applied z-normalization')

#plots.plotFeatures(D_train, L_train, utils.featuresNames, utils.classesNames)

#-------- Dimensionality analysis (PCA) --------#

#PCA.correlation_heatmap(D_train, L_train)

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
print('\t\t\tFull covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)

#Diag covariance
model_diagMVG.train(DTR, LTR)
scores = model_diagMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tDiag covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)

#Tied covariance
model_tiedMVG.train(DTR, LTR)
scores = model_tiedMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tTied covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=6)')

(DTR,LTR),(DTE,LTE) = utils.split_db_singleFold(D_train6, L_train)

#Full covariance
model_MVG.train(DTR, LTR)
scores = model_MVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tFull covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)

#Diag covariance
model_diagMVG.train(DTR, LTR)
scores = model_diagMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tDiag covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)

#Tied covariance
model_tiedMVG.train(DTR, LTR)
scores = model_tiedMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tTied covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=5)')

(DTR,LTR),(DTE,LTE) = utils.split_db_singleFold(D_train5, L_train)

#Full covariance
model_MVG.train(DTR, LTR)
scores = model_MVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tFull covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)

#Diag covariance
model_diagMVG.train(DTR, LTR)
scores = model_diagMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tDiag covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)

#Tied covariance
model_tiedMVG.train(DTR, LTR)
scores = model_tiedMVG.predictAndGetScores(DTE)
minDCF_05 = metrics.compute_minDCF(scores, LTE, 0.5, 1, 1)
minDCF_09 = metrics.compute_minDCF(scores, LTE, 0.9, 1, 1)
minDCF_01 = metrics.compute_minDCF(scores, LTE, 0.1, 1, 1)
print('\t\t\tTied covariance MVG --> ', minDCF_05, minDCF_09, minDCF_01)


print('\tK-Fold approach\n')

print('\t\tZ-norm | no PCA')

print('\t\t\tFull covariance MVG --> ', utils.KFold(D_train, L_train, GaussianClassifiers.GaussianClassifier()))
print('\t\t\tDiag covariance MVG --> ', utils.KFold(D_train, L_train, GaussianClassifiers.GaussianClassifier_NaiveBayes()))
print('\t\t\tTied covariance MVG --> ', utils.KFold(D_train, L_train, GaussianClassifiers.GaussianClassifier_TiedCovariances()))
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=7)')

print('\t\t\tFull covariance MVG --> ', utils.KFold(D_train7, L_train, GaussianClassifiers.GaussianClassifier()))
print('\t\t\tDiag covariance MVG --> ', utils.KFold(D_train7, L_train, GaussianClassifiers.GaussianClassifier_NaiveBayes()))
print('\t\t\tTied covariance MVG --> ', utils.KFold(D_train7, L_train, GaussianClassifiers.GaussianClassifier_TiedCovariances()))
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=6)')

print('\t\t\tFull covariance MVG --> ', utils.KFold(D_train6, L_train, GaussianClassifiers.GaussianClassifier()))
print('\t\t\tDiag covariance MVG --> ', utils.KFold(D_train6, L_train, GaussianClassifiers.GaussianClassifier_NaiveBayes()))
print('\t\t\tTied covariance MVG --> ', utils.KFold(D_train6, L_train, GaussianClassifiers.GaussianClassifier_TiedCovariances()))
#---------------------------------------------------------------------------------------------------------
print('\t\tZ-norm | PCA(m=5)')

print('\t\t\tFull covariance MVG --> ', utils.KFold(D_train5, L_train, GaussianClassifiers.GaussianClassifier()))
print('\t\t\tDiag covariance MVG --> ', utils.KFold(D_train5, L_train, GaussianClassifiers.GaussianClassifier_NaiveBayes()))
print('\t\t\tTied covariance MVG --> ', utils.KFold(D_train5, L_train, GaussianClassifiers.GaussianClassifier_TiedCovariances()))

#--- Logistic Regression ---#

print('Starting linear Logistic Regression analysis:\n')

# linear LR tuning
hyperparameter_tuning.linear_LR_tuning(D_train, L_train, mode = 'singleFold')
hyperparameter_tuning.linear_LR_tuning(D_train, L_train, mode = 'KFold')
hyperparameter_tuning.linear_LR_tuning(D_train7, L_train, mode = 'singleFold')
hyperparameter_tuning.linear_LR_tuning(D_train7, L_train, mode = 'KFold')

linear_LR_lambda = 1e-5
model_linearLR = LogisticRegression.LinearLR()

print('Selected value for lambda (linear LR): ', linear_LR_lambda)

print('\tK-Fold approach\n')

print('\t\tZ-norm | no PCA')
print('\t\t\tLinear LR (λ=1e-5) --> ', utils.KFoldLR(D_train, L_train, model_linearLR, linear_LR_lambda))
#---------------------------------------------------------------------------------------------------------

print('\t\tZ-norm | PCA(m=7)')
print('\t\t\tLinear LR (λ=1e-5) --> ', utils.KFoldLR(D_train7, L_train, model_linearLR, linear_LR_lambda))

      
print('Starting quadratic Logistic Regression analysis:\n')

# quadratic LR tuning
hyperparameter_tuning.quadratic_LR_tuning(D_train, L_train, mode='singleFold')
hyperparameter_tuning.quadratic_LR_tuning(D_train, L_train, mode='KFold')
hyperparameter_tuning.quadratic_LR_tuning(D_train7, L_train, mode='singleFold')
hyperparameter_tuning.quadratic_LR_tuning(D_train7, L_train, mode='KFold')


quadratic_LR_lambda = 1e-5
model_quadraticLR = LogisticRegression.QuadraticLR()

print('Selected value for lambda (quadratic LR): ', quadratic_LR_lambda)

print('\tK-Fold approach\n')

print('\t\tZ-norm | no PCA')
print('\t\t\tQuadratic LR (λ=1e-5) --> ', utils.KFoldLR(D_train, L_train, model_quadraticLR, quadratic_LR_lambda))
#---------------------------------------------------------------------------------------------------------

print('\t\tZ-norm | PCA(m=7)')
print('\t\t\tQuadratic LR (λ=1e-5) --> ', utils.KFoldLR(D_train7, L_train, model_quadraticLR, quadratic_LR_lambda))


print('Starting linear SVM analysis:\n')

hyperparameter_tuning.balanced_linear_SVM_tuning(D_train, L_train, mode='KFold')
hyperparameter_tuning.balanced_linear_SVM_tuning(D_train7, L_train, mode='KFold')

linear_SVM_C = 1e-3
print('Selected value for C (linear SVM): ', linear_SVM_C)

print('\tK-Fold approach\n')
print('\t\tZ-norm | no PCA')

model = SupportVectorMachines.BalancedLinearSVM()

print('Linear SVM (pi_t=0.9) --> ', utils.KFoldSVM(D_train, L_train, model, C=linear_SVM_C, K=3, prior=0.5, pi_T=0.9))
print('Linear SVM (pi_t=0.5) --> ', utils.KFoldSVM(D_train, L_train, model, C=linear_SVM_C, K=3, prior=0.5, pi_T=0.5))
print('Linear SVM (pi_t=0.1) --> ', utils.KFoldSVM(D_train, L_train, model, C=linear_SVM_C, K=3, prior=0.5, pi_T=0.1))

print('\t\tZ-norm | PCA(m=7)')

model = SupportVectorMachines.BalancedLinearSVM()

print('Linear SVM (pi_t=0.9) --> ', utils.KFoldSVM(D_train7, L_train, model, C=linear_SVM_C, K=3, prior=0.5, pi_T=0.9))
print('Linear SVM (pi_t=0.5) --> ', utils.KFoldSVM(D_train7, L_train, model, C=linear_SVM_C, K=3, prior=0.5, pi_T=0.5))
print('Linear SVM (pi_t=0.1) --> ', utils.KFoldSVM(D_train7, L_train, model, C=linear_SVM_C, K=3, prior=0.5, pi_T=0.1))


print('Starting polynomial kernel SVM analysis:\n')

# hyperparameter_tuning.balanced_poly_SVM_tuning(D_train, L_train, mode='KFold')
# hyperparameter_tuning.balanced_poly_SVM_tuning(D_train7, L_train, mode='KFold')

poly_SVM_c = 10
poly_SVM_C = 1e-5
print('Selected values for polynomial SVM --> C: ', poly_SVM_C, ' c: ', poly_SVM_c)

model = SupportVectorMachines.BalancedQuadraticSVM()
print('Polynomial SVM (C=1e-5, c=10) --> ', utils.KFoldSVM_kernel(D_train, L_train, model, kernel='poly', C=poly_SVM_C, K=3, c=poly_SVM_c))
print('Polynomial SVM (C=1e-5, c=10) --> ', utils.KFoldSVM_kernel(D_train7, L_train, model, kernel='poly', C=poly_SVM_C, K=3, c=poly_SVM_c))


print('Starting RBF kernel SVM analysis:\n')

hyperparameter_tuning.balanced_RBF_SVM_tuning(D_train, L_train, mode='KFold')
hyperparameter_tuning.balanced_RBF_SVM_tuning(D_train7, L_train, mode='KFold')


RBF_SVM_C = 1e-1
RBF_SVM_gamma = 1e-3

print('Selected values for RBF SVM --> C: ', RBF_SVM_C, ' gamma: ', RBF_SVM_gamma)

model = SupportVectorMachines.BalancedQuadraticSVM()
print('RBF kernel SVM (C=1e-1, gamma=1e-3) --> ', utils.KFoldSVM_kernel(D_train, L_train, model, kernel='RBF', C=RBF_SVM_C, K=3, gamma=RBF_SVM_gamma))
print('RBF kernel SVM (C=1e-1, gamma=1e-3) --> ', utils.KFoldSVM_kernel(D_train7, L_train, model, kernel='RBF', C=RBF_SVM_C, K=3, gamma=RBF_SVM_gamma))


print('Starting GMM analysis:\n')

hyperparameter_tuning.GMM_tuning(D_train, L_train, mode='KFold')
hyperparameter_tuning.GMM_tuning(D_train7, L_train, mode='KFold')
hyperparameter_tuning.diag_GMM_tuning(D_train, L_train, mode='KFold')
hyperparameter_tuning.diag_GMM_tuning(D_train7, L_train, mode='KFold')
hyperparameter_tuning.tied_GMM_tuning(D_train, L_train, mode='singleFold')
hyperparameter_tuning.tied_GMM_tuning(D_train7, L_train, mode='KFold')

GMM_M = 8
print('Selected value for M: ', GMM_M)

model_full = GaussianMixtureModels.GMM()
model_diag = GaussianMixtureModels.GMMDiag()
model_tied = GaussianMixtureModels.GMMTiedCov()
    
print('\tK-Fold approach\n')
print('\t\tZ-norm | no PCA')

print('Full-Cov GMM --> ', utils.KFoldGMM(D_train, L_train, model_full, K=3, M=GMM_M))
print('Diag-Cov GMM --> ', utils.KFoldGMM(D_train, L_train, model_diag, K=3, M=GMM_M))
print('Tied-Cov GMM --> ', utils.KFoldGMM(D_train, L_train, model_tied, K=3, M=GMM_M))

print('\t\tZ-norm | PCA(m=7)')

print('Full-Cov GMM --> ', utils.KFoldGMM(D_train7, L_train, model_full, K=3, M=GMM_M))
print('Diag-Cov GMM --> ', utils.KFoldGMM(D_train7, L_train, model_diag, K=3, M=GMM_M))
print('Tied-Cov GMM --> ', utils.KFoldGMM(D_train7, L_train, model_tied, K=3, M=GMM_M))
"""

#### ACTUAL DCF + SCORE CALIBRATION

print('\t\t\t\tTEST SET\n')

D_test, L_test = utils.load('data/Test.txt')
print('Test Dataset loaded\n')

#-------- Apply Z-normalization --------#

D_test, _, _ = utils.ZNormalization(D_test)
print('Applied z-normalization')

# PCA (m=7)
D_test7 = PCA.compute_PCA(D_test, 7)

print('Applied PCA with m=7\n')

print('Starting Gaussian Classifiers analysis:\n')

model_MVG = GaussianClassifiers.GaussianClassifier()
model_diagMVG = GaussianClassifiers.GaussianClassifier_NaiveBayes()
model_tiedMVG = GaussianClassifiers.GaussianClassifier_TiedCovariances()

print('\tK-Fold approach\n')

print('\t\tZ-norm | no PCA')

model_MVG.train(D_train, L_train)
model_diagMVG.train(D_train, L_train)
model_tiedMVG.train(D_train, L_train)

print('\t\t\tFull covariance MVG --> ', utils.KFold(D_test, L_test, model_MVG, train=False))
print('\t\t\tDiag covariance MVG --> ', utils.KFold(D_test, L_test, model_diagMVG, train=False))
print('\t\t\tTied covariance MVG --> ', utils.KFold(D_test, L_test, model_tiedMVG, train=False))

print('\t\tZ-norm | PCA(m=7)')

model_MVG.train(D_train7, L_train)
model_diagMVG.train(D_train7, L_train)
model_tiedMVG.train(D_train7, L_train)

print('\t\t\tFull covariance MVG --> ', utils.KFold(D_test7, L_test, model_MVG, train=False))
print('\t\t\tDiag covariance MVG --> ', utils.KFold(D_test7, L_test, model_diagMVG, train=False))
print('\t\t\tTied covariance MVG --> ', utils.KFold(D_test7, L_test, model_tiedMVG, train=False))

print('Starting linear Logistic Regression analysis:\n')

linear_LR_lambda = 1e-5
model_linearLR = LogisticRegression.LinearLR()

print('Selected value for lambda (linear LR): ', linear_LR_lambda)

print('\tK-Fold approach\n')

print('\t\tZ-norm | no PCA')
model_linearLR.train(D_train, L_train, linear_LR_lambda)
print('\t\t\tLinear LR (λ=1e-5) --> ', utils.KFoldLR(D_test, L_test, model_linearLR, linear_LR_lambda, train=False))
#---------------------------------------------------------------------------------------------------------

print('\t\tZ-norm | PCA(m=7)')
model_linearLR.train(D_train7, L_train, linear_LR_lambda)
print('\t\t\tLinear LR (λ=1e-5) --> ', utils.KFoldLR(D_test7, L_test, model_linearLR, linear_LR_lambda, train=False))


print('Starting quadratic Logistic Regression analysis:\n')

quadratic_LR_lambda = 1e-5
model_quadraticLR = LogisticRegression.QuadraticLR()

print('Selected value for lambda (quadratic LR): ', quadratic_LR_lambda)

print('\tK-Fold approach\n')

print('\t\tZ-norm | no PCA')
model_quadraticLR.train(D_train, L_train, quadratic_LR_lambda)
print('\t\t\tQuadratic LR (λ=1e-5) --> ', utils.KFoldLR(D_test, L_test, model_quadraticLR, quadratic_LR_lambda, train=False))
#---------------------------------------------------------------------------------------------------------

print('\t\tZ-norm | PCA(m=7)')
model_quadraticLR.train(D_train7, L_train, quadratic_LR_lambda)
print('\t\t\tQuadratic LR (λ=1e-5) --> ', utils.KFoldLR(D_test7, L_test, model_quadraticLR, quadratic_LR_lambda, train=False))




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
