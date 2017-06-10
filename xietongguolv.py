from math import sqrt
from numpy import *
import operator

from algroithms.kernels import cossim
from process.processmain import processReturn

def classifyperson(inputdata,data,lable,k):#传入测试数据，总数据，总数据类别，K个
    resultdata=[]#结果类别
    dataSetSize=data.shape[0]
    for indata in inputdata:
        sortdistance=[]
        for tran in data:
            simnum=cossim(indata,tran)##############修改函数
            sortdistance.append(simnum)
        sortdistance=array(sortdistance)
        sortdistance=argsort(-sortdistance)
        classcount={}
        for i in range(k):
             vote=lable[sortdistance[i]]
             classcount[vote] = classcount.get(vote,0)+1
        sortdic=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
        resultdata.append(sortdic[0][0])#返回最近的所有点排序，选第一个
    return resultdata

def knn(wordNum,pro,k1,k2):#knn运算，抽取的字数，训练比率
    returnlist, classifylist, doctorlist = processReturn(wordNum)
    a=int(len(classifylist)*pro)
    trainlable=array(classifylist[:a])
    traindata=array(returnlist[:a])
    goallable=classifylist[a:]
    goaldata=array(returnlist[a:])
    for num in range(k1,k2):
        right = 0
        result=classifyperson(goaldata,traindata,trainlable,num)
        for i in range(len(goallable)):
                 if goallable[i]==result[i]:
                         right=right+1
        print("第",num,"个的准确率： ",(right/len(goallable)))



knn(300,0.75,3,10)