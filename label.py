import math
import time

import jieba
import jieba.analyse
from termcolor import *


# from sklearn import feature_extraction
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# texts, classify, doctor = querySql()#返回三个列表
def dataDir(classify,doctor):#传入原始SQL数据

    c=0
    classifydir={}
    # classifydir1={}
    doctordir={}
    # doctordir1 = {}
    for i in classify :
        if i not in classifydir:
            classifydir[i]=c
            c+=1
    c1=0
    for i in doctor:
        if i not in doctordir:
            doctordir[i]=c1
            c1+=1
    return  classifydir,doctordir   #类别编号，医生编号
def createLabel(classify, doctor):#传入类别，医生
    classifydir, doctordir=dataDir( classify, doctor)
    print(colored("类别编号字典：     ","blue"),colored(classifydir,"blue"))
    print(colored("医生编号字典：     ","blue"),colored(doctordir,"blue"))
    classifylist=[]
    doctorlist=[]
    for i in classify:
        classifylist.append(classifydir[i])
    for i in doctor:
        doctorlist.append(doctordir[i])
    return classifylist,doctorlist #按顺序返回类别 ，医生编号

def splitLine(texts):#所有文档
    t0 = time.clock()
    jieba.load_userdict("D:\Pythonprogramme\yiliao\process\ciku.txt")
    stopwords=[]
    alllines=[]
    allwords=[]
    countnum=[]
    c=open("D:\Pythonprogramme\yiliao\process\stopword.txt").read().split()
    for i in c :
        stopwords.append(i)
    stopwords.append(' ')
    stopwords=set(stopwords)
    # print(stopwords)
    for text in texts:
        textlist = jieba.cut(text)
        line={}
        c=0
        for word in textlist:
            if word not in stopwords:
                c += 1
                if word not in line:
                    line[word]=1
                else:
                    line[word]+=1
                if word not in allwords:
                    allwords.append(word)
        countnum.append(c)
        alllines.append(line)
    print(colored("分词使用时间: ","red"),time.clock()-t0)
    return alllines,allwords,countnum   #返回所有文本字典，所有字典，每个文档的所有词数

#目前不用，用于存放在本地
def grade(alllines,allwords,countnum):#所有文本字典，所有字典，每个文档的所有词数
    idftxt=open("idf.txt","w")
    tfidftxt=open("tfidf.txt","w")
    num=len(alllines)#文档总数
    print("总文档数"+str(num))
    appearnum=[1]*len(allwords)#单词出现在文档的次数
    tfidflist=[] #最后所有比分TFIDF
    idf=[]#idf 值
    for line in alllines:#每一个文档
        for wordindex in range(len(allwords)):
                if allwords[wordindex] in line:
                    appearnum[wordindex]+=1#统计总出现
    for idfnum in appearnum:
        num1=math.log(num/idfnum)
        idf.append(num1)#计算
        idftxt.write(str(num1)+"&&")
    for li in range(len(alllines)):#遍历所有文档（字典）
        linum=[0]*len(allwords)#ALLWORDS向量
        for index in range(len(allwords)):#所有的ALLWORDS
            if allwords[index] in alllines[li]:#字典
                tfidfnum=int(alllines[li][allwords[index]])/countnum[li]*idf[index]
                linum[index] = tfidfnum
                tfidftxt.write(str(tfidfnum)+"&&")
            else:
                tfidftxt.write("0.0&&")
        tfidflist.append(linum)
        tfidftxt.write("$$")
    print("最后结果  "+str(len(tfidflist)))
    idftxt.close()
    tfidftxt.close()
# alllines,allwords,countnum=splitLine(texts)
# print(len(allwords))
# grade(alllines,allwords,countnum)


    










