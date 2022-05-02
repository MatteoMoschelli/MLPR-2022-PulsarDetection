# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:25:59 2022

@author: Utente
"""

import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# -------------------- FUNCTIONS FOR DIMENSIONALITY ANALYSIS ---------------------

# Projection matrix using PCA
def compute_PCA(data, m):
    C = utils.covMatrix(data)[1]

    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]

    # Projected points (matrix)
    DP = np.dot(P.T, data)
    return DP
    
# Print the correlation heatmaps associated with the entire dataset, samples of class 0 and samples of class 1
def correlation_heatmap(D, L):
    plt.figure()
    seaborn.heatmap(np.corrcoef(D), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==0]), linewidth=0.2, cmap="Reds", square=True,cbar=False)
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==1]), linewidth=0.2, cmap="Blues", square=True, cbar=False)
    return