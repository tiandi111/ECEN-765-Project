import numpy as np
import math
import matplotlib.pyplot as plt
import random
from DataProcessing import DataPro

class NBClassifier_mnist():

    def __init__(self, dataset, label, num_class, range, num_range):
        self.dataset = dataset
        self.label = label
        self.num_class = num_class
        self.range = range
        self.num_range = num_range
        self.PDigit_marginal = np.zeros( num_class )
        self.PPixel_marginial = np.ones( (len(dataset[0]),num_range) )/(len(dataset)+num_range) #* math.exp(-10000)#pixel[i] = the p of pixel_i = 0
        self.PPixel_conditional = np.zeros( num_class*len(dataset[0])*num_range) #* math.exp(-10000)
        self.PPixel_conditional = np.reshape( self.PPixel_conditional, (num_class, len(dataset[0]), num_range) )

    def Train( self ):
        label = self.label
        dataset = self.dataset
        num_range = self.num_range
        span = (self.range[1]-self.range[0]) / self.num_range
        lower = self.range[0]
        upper = self.range[1]
        PDigit_marginal = self.PDigit_marginal
        PPixel_marginial = self.PPixel_marginial
        PPixel_conditional = self.PPixel_conditional
        'First step: construct PDigit_marginal table'
        for i, item in enumerate( dataset ):
            PDigit_marginal[ int(label[i]) ] += 1/len( dataset )

        'Second step: construct PPixel_marginal table'
        for i, digit in enumerate( dataset ):
            for j, pixel in enumerate( dataset[i] ):
                for r in range(num_range):
                    if(( pixel >= lower+span*r )and( pixel < lower+span*(r+1))):
                        PPixel_marginial[j][r] += 1 / (len(dataset)+num_range)
                        #print(pixel, lower+span*r, lower+span*(r+1))

        'Third step: construct PPixel_conditional table'
        for i, item in enumerate( dataset ):
            for j, pixel in enumerate( dataset[i] ):
                for r in range(num_range):
                    if(( pixel >= lower+span*r )and( pixel <= lower+span*(r+1))):
                        PPixel_conditional[ int(label[i]) ][ j ][r] += 1 / (PDigit_marginal[ int(label[i]) ]*len(dataset)+num_range)
        for i in range(10):
            for j in range( len(dataset[0]) ):
                for r in range(num_range):
                    PPixel_conditional[i][j][r] += 1 / (PDigit_marginal[i]*len(dataset) + num_range)
        #for i in range(0,10):
         #   i = random.randint(0,9)
          #  j = random.randint(0,len(dataset[i])-1)
           # print(PPixel_conditional[i][j],i,j)
        self.PPixel_conditional = PPixel_conditional
        self.PDigit_marginal = PDigit_marginal
        self.PPixel_marginial = PPixel_marginial


    def Predict(self, testset, testlabel):
        PDigit_marginal = self.PDigit_marginal
        PPixel_marginial = self.PPixel_marginial
        PPixel_conditional = self.PPixel_conditional
        num_range = self.num_range
        lower, upper = self.range
        span = (self.range[1] - self.range[0]) / self.num_range
        predictlabel = []
        accuracy = 0

        'calculate PPixel_mariginal_final and PPixel_conditional_final'
        for i, image in enumerate(testset):
            Pcmp = [0, 0]  # first element is P, second element is digit
            for k in range(self.num_class):
                PPixel_marginal_final = 1
                PPixel_conditional_final = 1
                for j, pixel in enumerate(testset[i]):
                    for r in range(num_range):
                        if ((pixel >= lower + span * r) and (pixel < lower + span * (r + 1))):
                            PPixel_marginal_final *= PPixel_marginial[j][r]
                            PPixel_conditional_final *= PPixel_conditional[k][j][r]
                P = PPixel_conditional_final * PDigit_marginal[k] / PPixel_marginal_final
                #print(i,k,P, PPixel_conditional_final, PDigit_marginal[k], PPixel_marginal_final)
                if( P > Pcmp[0] ):
                    Pcmp[0] = P
                    Pcmp[1] = k
            predictlabel.append( Pcmp[1] )
            if( Pcmp[1] == testlabel[i] ):
                accuracy += 1 / len(testset)
            #print('for the', i, 'th test point, prediction is: ', Pcmp[1], ', real label is: ', testlabel[i])
        return accuracy*100

if __name__ == '__main__':
    Data = DataPro(hello='hi')
    trainset, trainlabel = Data.GetExcel('mnist_train_.xlsx', 0, 60000)
    testset, testlabel = Data.GetExcel('mnist_test_.xlsx', 0, 10000)
    'Test on test set with k from 1 to 10'
    for i in range (1,10):
        nbc = NBClassifier_mnist(trainset, trainlabel, 10, [0, 255], i*25)
        nbc.Train()
        er = nbc.Predict(testset, testlabel)
        print('when i =', i, ', the accuracy on testset is ', er, '%')
