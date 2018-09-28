from DataProcessing import DataPro
import matplotlib.pyplot as plt
import numpy as np

class KNN():
    pass

class node( object ):
    def __init__(self, dataset=None, index=[0], range=0, splitD=0, label=None, leftnode=None, rightnode=None, leaf = False):
        self.index = index
        self.range = range
        self.splitD = splitD
        self.leftnode = leftnode
        self.rightnode = rightnode
        self.dataset = dataset
        self.leaf = leaf
        self.label = label
        if (dataset):
            self.num = len(self.dataset)
            self.dimension = len(self.dataset[0])
        else:
            self.num = 0
            self.dimension = 0
        self.dis = 0
        self.nearest = 0

    def KDTree_Basic(self, p ):
        num = self.num
        maxsplit = len( self.dataset[0] )
        if( p.index+1 == num):
            return p
        else:
            if( self.dataset[p.index+1][p.splitD] <= self.dataset[p.index][p.splitD] ):
                if (p.splitD == maxsplit-1):
                    split = 0
                else:
                    split = p.splitD+1
                leftnode = self.addleftnode( p, split )
                print(leftnode.index,0)
                self.KDTree( leftnode )

            else:
                if (p.splitD == maxsplit-1):
                    split = 0
                else:
                    split = p.splitD+1
                rightnode = self.addrightnode( p, split)
                print(0,rightnode.index)
                self.KDTree( rightnode )

    def addleftnode(self, Father, splitD):
        p = node( self.dataset, Father.index+1, 0, splitD, None, None)
        Father.leftnode = p
        return p

    def addrightnode(self, Father, splitD):
        p = node( self.dataset, Father.index+1, 0, splitD, None, None)
        Father.rightnode = p
        return p

    def KDTree(self,dataset, label, split=0, counts=0):
        kd = node()
        if( len(dataset)==0 ):
            return kd
        d = self.Partition(dataset, split, 0, len(dataset)-1)
        if (split == len(dataset[0])-1):
            split = 0
        else:
            split = split+1
        domain = dataset[d]
        kd.label = label[d]
        #r = range vector of ex
        subset, sublabel = self.remove(dataset, label, d)
        if ( len(subset) == 0 ):
            kd.leaf = True
        subset_left, subset_right, sublabel_left, sublabel_right = self.setsplit(subset, label, d)
        kdleft = self.KDTree( subset_left, sublabel_left, split, counts )
        kdright = self.KDTree( subset_right, sublabel_right, split, counts )
        kd.index = domain
        kd.splitD = split
        kd.leftnode = kdleft
        kd.rightnode = kdright
        return kd

    def SearchKDTree(self, kd, target, hr, maxdis, k, node):
        if (kd == None):
            dis = node.dis
            return kd,dis,node
        elif(len(kd.index) == 1):
            dis = node.dis
            return kd,dis,node
        s = kd.splitD #split field
        pivot = kd.index
        if ((target[s] <= pivot[s])):
            nearer = kd.leftnode; nehr = 0; maxdis = self.Caldis(target, pivot)
            further = kd.rightnode; fuhr = pivot[s]-target[s]
        elif((target[s] > pivot[s])):
            nearer = kd.rightnode; nehr = 0; maxdis = self.Caldis(target, pivot)
            further = kd.leftnode; fuhr = target[s]-pivot[s]
        nearest, dis, node = self.SearchKDTree( nearer, target, nehr, maxdis, k, node)
        maxdis = min(maxdis, dis)
        if(dis>maxdis):
            dis = maxdis
            nearest = kd
        if  ( fuhr < (maxdis**0.5) ) :
            if ( self.Caldis(further.index,target) < dis ):
                nearest = further
                dis = self.Caldis(further.index,target)
                maxdis = dis
            tempnerest, tempdis, node = self.SearchKDTree( further, target, fuhr, maxdis, k, node )
            if ( tempdis < dis ):
                nearest = tempnerest; dis = tempdis
                if(node.size<k):
                    node = node.addinorder( nearest, dis)
                else:
                    node = node.addmiddle( nearest, dis)
                dis = node.dis
                return nearest, dis, node
        if (node.size < k):
            node = node.addinorder( nearest, dis)
        else:
            node = node.addmiddle( nearest, dis)
        dis = node.dis
        return nearest, dis, node

    def Caldis(self, a, b):
        d = np.multiply(a-b,a-b).sum()
        return d



    def Partition(self, dataset, dim, left, right ):
        tail = 0
        while(tail != len(dataset)//2):
            key = right; lastleft = left; lastright = right
            while (left < right):
                while ((left < right) and (dataset[left][dim] <= dataset[key][dim])):
                    left = left + 1
                while ((left < right) and (dataset[right][dim] >= dataset[key][dim])):
                    right = right - 1
                self.swap(dataset, left, right)
            self.swap(dataset, left, key)
            tail = left
            if (left > len(dataset) // 2):
                right = left - 1
                left = lastleft
            elif(left < len(dataset)//2 ):
                left = left + 1;
                right = lastright
        return left

    def remove(self, dataset, label, d):
        label.pop(d)
        dataset.pop(d)
        return dataset, label

    def setsplit(self, dataset, label, medium):
        rightset = []; rightlabel = []
        if (dataset == None):
            return rightset, rightset, rightlabel, rightlabel
        leftset = dataset[0:medium]
        leftlabel = label[0:medium]
        for i in range(medium, len(dataset)):
            rightlabel.append(label[i])
            rightset.append(dataset[i])
        return leftset, rightset, leftlabel, rightlabel

    def swap(self, dataset, a, b):
        x = dataset[a]
        dataset[a] = dataset[b]
        dataset[b] = x

class LList():

    def __init__(self, item=0, dis=0, next=None):
        self.item = item
        self.dis = dis
        self.next = next
        self.size = 1

    def addfirst(self, item, dis):
        a = LList(item, dis)
        a.next = self
        self = a
        self.size = self.size+1
        return self

    def addlast(self, item, dis):
        a = LList(item, dis)
        p = self
        while( p.next.next != None ):
            p = p.next
        p.next = a
        self.size = self.size+1
        return self

    def addmiddle(self, item, dis):
        a = LList(item, dis)
        p = self
        if(p.next == None):
            p.next = a;
            p.next.size = self.size
            self = self.next
            return self
        while(a.dis < p.next.dis ):
            p = p.next
            if(p.next==None):
                p.next = a
                self.next.size = self.size
                self = self.next
                return self
        a.next = p.next
        p.next = a
        self.next.size = self.size
        self = self.next
        return(self)

    def addinorder(self, item, dis):
        a = LList(item, dis)
        p = self
        if(dis>self.dis):
            node = self.addfirst(self, item, dis)
            self.size = self.size+1
            return self
        else:
            if(p.next == None):
                p.next = a
                self.size = self.size+1
                return self
            while((a.dis<=p.next.dis)and(p.next!=None)):
                p = p.next
                if(p.next == None):
                    p.next = a
                    self.size = self.size + 1
                    return self
            a.next = p.next
            p.next = a
            self.size = self.size+1
            return self

    def deletefirst(self):
        self.size = self.size-1
        self = self.next
        return self

if __name__ == "__main__":
    errorplot = []
    k = []
    dataset = ([[1,4,6],[2,6,4],[5,4,5],[0,8,6],[3,6,8],[2,5,9],[5,8,0]]) #just some random test sample
    KDT = node(dataset)
    Data = DataPro(hello='hi')
    testset1=[]
    label1=[]
    trainset, trainlabel = Data.Getdata("trainingDigits")
    testset, testlabel = Data.Getdata("testDigits")
    'Test on test set with k from 1 to 10'
    '''
    for o in range(1,20):
        accuracy = 0;
        volume = len(testset)
        for i, item in enumerate(testset):
            queue = LList([0], 9999999)
            vote = np.zeros(10)
            a, b, node = KDT.SearchKDTree(kd, testset[i], 0, 100000000, k=o, node=queue)
            while ((node != None)and(node.item.label !=None)):
                vote[int(node.item.label)] = vote[int(node.item.label)] + 1
                node = node.next
            for j, item in enumerate(vote):
                if (max(vote) == item):
                    predict = j
                    break
            if (predict == testlabel[i]):
                accuracy = accuracy + 1 / len(testset)
            #print('for the ',i ,'th test point, prediction is:', predict, ' real label is: ', testlabel[i])
        print('when k=',o,',accuracy on testset:', accuracy * 100, '%')
        errorplot.append((1-accuracy)*100)
        k.append(o)
    plt.figure()
    plt.plot(k, errorplot, color='red', label='test error rate')
    plt.xlabel('k')
    plt.ylabel('error rate %')
    plt.xticks(np.linspace(0, 11, 12))
    plt.title('Error rate with different k, KNN')
    plt.show()
    '''
    'Train KNN with samples increasing from 100 to 1800 and plot the result'
    x=[]
    xaxis_samplenum = []
    er_test = []
    er_train = []
    for i in range(1, 19):
        xaxis_samplenum.append(i * 100)
        newset, newlabel = Data.Partition(i * 100, trainset, trainlabel)
        accuracy = 0;
        volume = len(testset)
        print(type(node()))
        KDT = node(dataset)
        x.append(KDT.KDTree(newset, label=newlabel, counts=1))
        for j in range(2):
            accuracy=0;
            if(j==0):
                testset1 = newset
                label1 = newlabel
            else:
                testset1 = testset
                label1 = testlabel
            for r, item in enumerate(testset1):
                queue = LList([0], 9999999)
                vote = np.zeros(10)
                a, b, nodes = KDT.SearchKDTree(x[i-1], testset1[r], 0, 100000000, k=1, node=queue)
                while ((nodes != None) and (nodes.item.label != None)):
                    vote[int(nodes.item.label)] = vote[int(nodes.item.label)] + 1
                    nodes = nodes.next
                for p, item in enumerate(vote):
                    if (max(vote) == item):
                        predict = p
                        break
                if (predict == label1[r]):
                    accuracy = accuracy + 1 / len(label1)
                # print('for the ',i ,'th test point, prediction is:', predict, ' real label is: ', testlabel[i])
            if(j==0): er_train.append((1-accuracy)*100)
            else: er_test.append((1-accuracy)*100)
            print('when k=', 1, ',accuracy on testset:', accuracy * 100, '%')
           #errorplot.append((1 - accuracy) * 100)
    plt.figure()
    plt.plot(xaxis_samplenum, er_test, color='red', label='test error rate')
    plt.plot(xaxis_samplenum, er_train, color='blue', label='train error rate')
    plt.xlabel('training sample volume')
    plt.ylabel('error rate %')
    plt.xticks(np.linspace(0, 1900, 20), rotation=45)
    plt.title('Error rate with different training volume, KNN')
    plt.legend()
    plt.show()