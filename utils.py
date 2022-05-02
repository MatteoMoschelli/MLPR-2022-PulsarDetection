# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:47:32 2022

@author: Utente
"""

import numpy as np
from scipy.stats import norm
import metrics

# -------------------- CONSTANTS ---------------------

classesNames = []
featuresNames = []

# ----------------- UTILITY FUNCTIONS -----------------

# Auxiliary function to transform 1-dim vectors to column vectors.
def vcol(v):
    return v.reshape((v.size, 1))

# Auxiliary function to transform 1-dim vectors to row vectors.
def vrow(v):
    return (v.reshape(1, v.size))

def load(filename):
    samples_list = []
    labels_list = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(',')
            if data[0] != '\n':
                for i in range(len(data)-1):
                    data[i] = float(data[i])
                data[-1] = int(data[-1].rstrip('\n'))
                # Now create a 1-dim array and reshape it as a column vector,
                # then append it to the appropriate list
                samples_list.append(vcol(np.array(data[0:-1])))
                # Append the value of the class to the appropriate list
                labels_list.append(data[-1])
    # We have column vectors, we need to create a matrix, so we have to
    # stack horizontally all the column vectors
    dataset = np.hstack(samples_list[:])
    # Create a 1-dim array with class labels
    labels = np.array(labels_list)
    return dataset, labels

# Used to split the input dataset into a training subset (80% of original data) and validation subset (20% of the original data)
def split_db_singleFold(D, L, seed=0):
    nTrain = int(D.shape[1]*8.0/10.0) #take 80% of the original dataset as training data
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) #random order for data indexes
    idxTrain = idx[0:nTrain] #nTrain indexes are for Training
    idxTest = idx[nTrain:] #others for Testing
    DTR = D[:, idxTrain] #training data
    DTE = D[:, idxTest] #evaluation data
    LTR = L[idxTrain] #training labels
    LTE = L[idxTest] #evaluation labels
    return (DTR, LTR), (DTE, LTE)

# Center some data by subtracting the mean
def centerData(D):
    means = D.mean(axis=1)
    means = vcol(means)
    centeredData = D - means
    return centeredData

# Compute Centered Dataset, Covariance Matrix
def covMatrix(Data):
    DC = centerData(Data)
    C = (1/DC.shape[1]) * np.dot(DC, DC.T)
    return DC, C

# Apply Gaussianization to D, using ranks based on training data TD
def Gaussianization(TD, D):
    if (TD.shape[0]!=D.shape[0]):
        print("Datasets not aligned in dimensions")
    ranks = []
    for j in range(D.shape[0]):
        tempSum=0
        for i in range(TD.shape[1]):
            tempSum += (D[j, :] < TD[j, i]).astype(int)
        tempSum += 1
        ranks.append(tempSum / (TD.shape[1] + 2))
    y = norm.ppf(ranks)
    return y

# Split the input dataset and the corresponding labels in K folds
def split_db_KFold(D, L, seed=0, K=5):
    folds = []
    labels = []
    
    N = int(D.shape[1]/K)
    
    # Generate a random seed
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    
    for i in range(K):
        folds.append(D[:,idx[(i * N): ((i + 1) * N)]])
        labels.append(L[idx[(i * N): ((i + 1) * N)]])
        
    return folds, labels

# Apply KFold cross-validation on the input model with the input dataset
def KFold(D, L, model, K=5, prior=0.5):
    if (K>1):
        folds, labels = split_db_KFold(D, L, seed=0, K=K)
        
        LTE = []
        scores = []
        
        for i in range(K):
            DTR = []
            LTR = []
            
            for j in range(K):
                if j!=i:
                    DTR.append(folds[j])
                    LTR.append(labels[j])
            DTE = folds[i]
            LTE.append(labels[i])
            DTR = np.hstack(DTR)
            LTR = np.hstack(LTR)
            model.train(DTR, LTR)
            scores.append(model.predictAndGetScores(DTE))
        
        scores = np.hstack(scores)
        LTE = np.hstack(LTE)
        labels = np.hstack(labels)
        
        return metrics.compute_minDCF(scores, LTE, prior, 1, 1)
    else:
        print("K cannot be <=1")
    return

