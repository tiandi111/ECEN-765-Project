import numpy as np
import math
import matplotlib.pyplot as plt
from DataProcessing import DataPro

class NBClassifier():

    def __init__(self, dataset, label, num_class):
        self.dataset = dataset
        self.label = label
        self.num_class = num_class
        self.PDigit_marginal = np.zeros( num_class )
        self.PPixel_marginial = np.ones( len(dataset[0]) ) / (len(dataset)+2)#pixel[i] = the p of pixel_i = 0
        self.PPixel_conditional = np.zeros( (num_class, len(dataset[0])) )

    def Train( self ):
        label = self.label
        dataset = self.dataset
        PDigit_marginal = self.PDigit_marginal
        PPixel_marginial = self.PPixel_marginial
        PPixel_conditional = self.PPixel_conditional
        'First step: construct PDigit_marginal table'
        for i, item in enumerate( dataset ):
            PDigit_marginal[ label[i] ] += 1/len( dataset )

        'Second step: construct PPixel_marginal table'
        for i, digit in enumerate( dataset ):
            for j, pixel in enumerate( dataset[i] ):
                if( pixel == 0 ):
                    PPixel_marginial[j] += 1 / (len(dataset)+2)

        'Third step: construct PPixel_conditional table'
        for i, item in enumerate( dataset ):
            for j, pixel in enumerate( dataset[i] ):
                if( pixel == 0 ):
                    PPixel_conditional[ label[i] ][ j ] += 1 / ((PDigit_marginal[ label[i] ] * len(dataset))+2)
        for i in range(9):
            for j in range(len(dataset[0])-1):
                PPixel_conditional[i][j] += 1 / ((PDigit_marginal[i] * len(dataset))+2)

    def Predict(self, testset, testlabel):
        PDigit_marginal = self.PDigit_marginal
        PPixel_marginial = self.PPixel_marginial
        PPixel_conditional = self.PPixel_conditional
        predictlabel = []
        accuracy = 0

        'calculate PPixel_mariginal_final and PPixel_conditional_final'
        for i, image in enumerate(testset):
            Pcmp = [0, 0]  # first element is P, second element is digit
            for k in range(self.num_class):
                PPixel_marginal_final = 1
                PPixel_conditional_final = 1
                for j, pixel in enumerate(testset[i]):
                    if (pixel == 0):
                        PPixel_marginal_final *= PPixel_marginial[j]
                        PPixel_conditional_final *= PPixel_conditional[k][j]
                    else:
                        PPixel_marginal_final *= 1 - PPixel_marginial[j]
                        PPixel_conditional_final *= 1 - PPixel_conditional[k][j]
                P = PPixel_conditional_final * PDigit_marginal[k] / PPixel_marginal_final
                #print(i,k,P, PPixel_conditional_final, PDigit_marginal[k], PPixel_marginal_final)
                if( P > Pcmp[0] ):
                    Pcmp[0] = P
                    Pcmp[1] = k
            predictlabel.append( Pcmp[1] )
            if( Pcmp[1] == testlabel[i] ):
                accuracy += 1 / len(testset)
            #print('for the', i, 'th test point, prediction is: ', Pcmp[1], ', real label is: ', testlabel[i])
        return 100-accuracy*100

if __name__ == '__main__':
    er_train = []
    er_test = []
    xaxis_samplenum = []
    Data = DataPro(hello='hi')
    trainset, trainlabel = Data.Getdata("trainingDigits")
    testset, testlabel = Data.Getdata("testDigits")
    'Train the model by training dataset with 100, 200, ... 1800 samples and plot the result'
    for i in range(1,19):
        xaxis_samplenum.append( i*100 )
        newset, newlabel = Data.Partition(i * 100, trainset, trainlabel)
        NBC = NBClassifier(newset, newlabel, 10)
        NBC.Train()
        print('when training sample is ', i*100)
        errorrate_train = NBC.Predict(newset, newlabel)
        er_train.append( errorrate_train )
        print('The error rate on training set is: ', errorrate_train, '%')
        errorrate_test = NBC.Predict(testset, testlabel)
        er_test.append( errorrate_test )
        print('The error rate on test set is: ', errorrate_test, '%')
    plt.figure()
    plt.plot(xaxis_samplenum, er_test, color = 'red', label = 'test error rate')
    plt.plot(xaxis_samplenum, er_train, color = 'blue', label = 'train error rate')
    plt.xlabel('training sample volume')
    plt.ylabel('error rate %')
    plt.xticks(np.linspace(0,1900,20),rotation = 45)
    plt.title('Error rate with different training volume, Naive Bayes Classifier')
    plt.show()


