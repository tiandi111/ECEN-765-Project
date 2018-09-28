#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import time
from DataProcessing import DataPro
from Kmeans import Kmeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

if __name__ == '__main__':
    'Display digits'
    time_start = time.time()
    Data = DataPro(hello = 'hi')
    Data.Display( "Display" )
    trainset, trainlabel = Data.Getdata("trainingDigits")
    testset, testlabel = Data.Getdata("testDigits")
    k = []; errorplot=[]
    'Test the KNN classifier from sklearn package'
    for i in range (1,11):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(trainset, trainlabel)
        prelabel = neigh.predict(testset)
        k.append(i)
        errorplot.append((1-neigh.score(testset, testlabel))*100)
        print(errorplot[i-1])
    plt.figure()
    plt.plot(k, errorplot, color='red', label='test error rate')
    plt.xlabel('k')
    plt.ylabel('error rate %')
    plt.xticks(np.linspace(0, 11, 12))
    plt.title('Error rate with different k, KNN')
    plt.show()




