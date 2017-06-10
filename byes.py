from numpy import *

from process.processmain import processReturn
from termcolor import *

def  byes(trainlist,trainlabel,all):
    trainlist=array(trainlist)
    alllabel=len(trainlabel)
    alllist=[0]*all
    for i in trainlabel:
        alllist[i]+=1
    for i in range(len(alllist)) :
        alllist[i]=alllist[i]/alllabel#每个类出现的概率
    numTraindoc=len(trainlist)#总共多少个文档
    numwords=len(trainlist[0])#总共多少个单词
    allbyes=[ones(numwords)]*all#array  a=[ones(3)]*5
    allscore=[2.0]*all#list
    for index in range(numTraindoc):
        allbyes[trainlabel[index]]+=trainlist[index]#每个文档对应类别，所有词的相叠加
        allscore[trainlabel[index]]+=sum(trainlist[index])#总数
    for i in range(all):
        allbyes[i]=log(allbyes[i]/allscore[i])
    return allbyes,alllist#每个类别所有单词出现的概率，每个类别的概率

def classiftByes(testlist,allbyes,alllist):
    testlist=array(testlist)
    result=[]
    for wordlist in testlist:
        prosmax= 0
        maxclassify=0
        for byeindex in range(len(allbyes)):
            if alllist[byeindex]!=0:
                pro =sum(wordlist * allbyes[byeindex])+log(alllist[byeindex])
                if pro> prosmax:
                    prosmax=pro
                    maxclassify=byeindex
        result.append(maxclassify)
    return  result

def byesRight(trainlist,trainlabel,testlist,testlabel,all):
    allbyes, alllist=byes(trainlist, trainlabel, all)
    results=classiftByes(testlist,allbyes,alllist)
    right=0
    for i in range(len(testlabel)):
        if testlabel[i]==results[i]:
                right+=1
    rightper=right/len(testlabel)
    return rightper


def mainByes(wordsnum,proportion):
    dataSet, classify, doctorlist = processReturn(wordsnum)
    all = len(set(classify))
    a = int(len(classify) * proportion)
    trandata = dataSet[:a]
    tranClassify = classify[:a]
    testdata = dataSet[a:]
    testClassify = classify[a:]
    rightper=byesRight(trandata, tranClassify, testdata, testClassify, all)
    print(colored("贝叶斯最后的结果是:     ","yellow"),rightper)



mainByes(100,0.75)


