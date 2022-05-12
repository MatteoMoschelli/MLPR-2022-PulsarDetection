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


D, L = utils.load('../WineQualityDetection/Train.txt')
D_gauss = utils.Gaussianization(D,D)


D = PCA.compute_PCA(D_gauss,10)

hyperparameter_tuning.linear_LR_tuning(D, L, mode = 'singleFold')