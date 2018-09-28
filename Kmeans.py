import numpy as np
from numpy import random

class Kmeans():

    def __init__(self, clusters):
        self.clusters = clusters
        self.finalcentroids = []

    def TrainKM(self, dataset, iter):
        centroids = []
        if (len(dataset) == 0):
            print('dataset is empty!')
        temlabel = self.InitLabel()#initiate label
        centroids = self.InitCentroid( shape = dataset[0].shape, cluster_num=self.clusters, value=False )#initiate centroid
        for counts in range(iter):
            #print("iterative counts: {}".format(counts))
            for k, item in enumerate(dataset):#Classify
                result = self.Classify(centroids, item )#get label and minimum distance
                temlabel[result[0]].append( k )#save classification table
            for i, table in enumerate(temlabel):#calculate centroids
                centroids[i] = self.CalCentroid(table, dataset)
        self.finalcentroids = centroids
        #print(self.finalcentroids)

    #Prediction
    def Predict(self, dataset, testlabel):
        label = []
        accuracy = 0
        reallabel = self.LabelCorrect(dataset, testlabel)
        for k, item in enumerate(dataset):
            result = self.Classify(self.finalcentroids, item )
            label.append(reallabel[result[0]])
        for i, item in enumerate(label):
            if(item == testlabel[i]):
                accuracy = accuracy+1
        print("cluster_num=",self.clusters,", Prediction accuracy on testset is:{}%".format(100*accuracy/len(label)))

    def Classify(self, centroids, datapoint):
        for i, centers in enumerate(centroids):
            if (i == 0):
                min = self.CalDistance(centers, datapoint)
                label = 0
            else:
                if (self.CalDistance(centers, datapoint)) < min:
                    min = self.CalDistance(centers, datapoint)
                    label = i
        return label, min

    #Initiate empty label
    def InitLabel(self):
        i = 0; label = []
        while( i<self.clusters ):
            x = []
            label.append(x)
            i = i+1
        return label

    #Randomly pick centroids
    def InitCentroid(self, shape, cluster_num, value):
        centroid = []
        for i in range( cluster_num ):
            if(value == True):
                centroid.append( np.zeros( shape ) )
            else:
                centroid.append( random.random( shape ) )
        return centroid

    #Calculate Euclid distance
    def CalDistance(self, a, b):
        try:
            distance = np.multiply(a-b,a-b)
        except:
            print("There is an error in distance calculation!")
        else:
            return distance.sum()

    def CalCentroid(self, index, dataset):
        x = np.zeros_like( dataset[0] )
        for i, k in enumerate(index):
            x = x + dataset[k]
        x = x / len(index)
        return x

    def LabelCorrect(self, dataset, label):
        Centersum = self.InitCentroid(shape=dataset[0].shape, cluster_num=10, value=True)
        count = np.zeros(10)
        reallabel = []
        for j, k in enumerate(label):
            count[k] = count[k] + 1
            Centersum[k] =  Centersum[k] + dataset[j]
        for i in range(10):
            Centersum[i] = Centersum[i] / count[i]
        for i, item in enumerate(self.finalcentroids):
            x = self.Classify(Centersum, item)
            reallabel.append(x[0])
        return reallabel
