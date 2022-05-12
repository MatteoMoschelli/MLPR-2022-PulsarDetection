# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:46:13 2022

@author: Utente
"""
import utils
import plots
import PCA
import metrics
import GaussianClassifiers


# load datasets

D_train, L_train = utils.load('data/Train.txt')

# preprocessing (gaussianization/Z-normalization)
# dimensionality analysis (PCA)
# training set analysis
## minDCF
#### MVG  -----> ok GaussianClassifiers
#### LR (linear + quad)
#### SVM (linear + polynomial & RBF kernel)
#### GMM
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
