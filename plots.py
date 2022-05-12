# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:57:54 2022

@author: Utente
"""
import matplotlib.pyplot as plt
import seaborn
import numpy as np

# -------------------- FUNCTIONS TO SHOW PLOTS ---------------------

def custom_hist(attr_index, xlabel, D, L, classesNames):
    # Function used to plot histograms. It receives the index of the attribute to plot,
    # the label for the x axis, the dataset matrix D, the array L with the values
    # for the classes and the list of classes names (used for the legend)
    plt.hist(D[attr_index, L == 0], color="#1e90ff",bins = 30,
             ec="#0000ff", density=True, alpha=0.6)
    plt.hist(D[attr_index, L == 1], color="#ff8c00",bins = 30,
             ec="#d2691e", density=True, alpha=0.6)
    plt.legend(classesNames)
    plt.xlabel(xlabel)
    plt.show()
    return

def plotFeatures(D, L, featuresNames, classesNames):
    for i in range(D.shape[0]):
        custom_hist(i, featuresNames[i], D, L, classesNames)


def plotDCF(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.9', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return

def heatmap(D, L):
    plt.figure()
    seaborn.heatmap(np.corrcoef(D), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==0]), linewidth=0.2, cmap="Reds", square=True,cbar=False)
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==1]), linewidth=0.2, cmap="Blues", square=True, cbar=False)
    return

def plotDCF(x, y, x_label, y_label):
    plt.figure()
    plt.plot(x, y[0], label='minDCF (pi=0.5)', color='r')
    plt.plot(x, y[1], label='minDCF (pi=0.9)', color='b')
    plt.plot(x, y[2], label='minDCF (pi=0.1)', color='g')
    plt.xscale("log")
    plt.xlim([min(x), max(x)])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(["minDCF (pi=0.5)", "minDCF (pi=0.9)", "minDCF (pi=0.1)"])
