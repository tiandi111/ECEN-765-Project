import os
import numpy as np
from PIL import Image
import xlrd

class DataPro():

    'This is constructor'
    def __init__(self, hello):
        self.hello = hello

    'Display all digit files under given directory as binary image'
    def Display(self, file):
        'return the path of each fiels under given directory'
        dataset = self.Getdata(file)[0]
        label = self.Getdata(file)[1]
        dataset = np.reshape( dataset, (32*len(dataset),32) )
        img = Image.fromarray(dataset * 255)
        img.show()
        print(dataset)
        print(label)

    def Getdata(self, file):
        dataset = []
        label = []
        for dirpath, dirnames, filenames in os.walk( file ):
            for filepath in filenames:
                file = os.path.join(dirpath, filepath)
                'read files'
                with open(file) as file_object:
                    for i in range(len(file)-1):
                        if(file[len(file)-1-i] == "_"):
                            label.append(int(file[len(file)-i-2]))
                    content = file_object.read()
                    digit = []
                    'convert str type to array'
                    for str in content:
                        if(str != '\n'):
                            digit.append(int(str))
                    digit = np.array(digit)
                    'convert to 32*32 format and make pixel with value 1 display white'
                    #digit = np.reshape(digit, (32,32))
                    dataset.append(digit)
        return dataset, label

    def GetExcel(self, file, start, end):
        data = []; label = []
        book = xlrd.open_workbook(file)
        table = book.sheets()[0]  # 通过索引顺序获取
        #table = book.sheet_by_index(0)  # 通过索引顺序获取
        #table = book.sheet_by_name(u'Sheet1')  # 通过名称获取
        nrows = table.nrows
        for i in range (start, end):
            data.append(np.array(table.row_values(i)[1:]))
            label.append(table.row_values(i)[0])
        print('数据准备完毕')
        return data, label

    def Partition(self, num,  trainset, trainlabel):
        newset_index = []
        newset = []
        newlabel = []
        num_class = np.zeros((10, 2))
        for j, label in enumerate(trainlabel):
            num_class[label][1] += 1
        for i in range(1,10):
            num_class[i][0] = num_class[i-1][1] + 1
            num_class[i][1] = num_class[i][1] + num_class[i-1][1]
        for i in range(10):
            for j in range( num//10 ):
                newset_index.append( np.random.randint(num_class[i][0], num_class[i][1]) )
        for i, index in enumerate( newset_index ):
            newset.append( trainset[ index ] )
            newlabel.append( trainlabel[ index ] )
        return newset, newlabel