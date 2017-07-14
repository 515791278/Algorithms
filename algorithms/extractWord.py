from process.sql import *

from process.label import *


def extract(alllines,allwords,countnum,k):#所有文本字典，所有词，每个文档的所有词数,提取个数
    t0 = time.clock()
    resultalllines=[]
    resultallwords=[]
    num=len(alllines)#文档总数
    resultcountnum = [k]*num
    print("文档总数：   "+str(num))
    appearnum=[1]*len(allwords)#单词出现在文档的次数
    tfidflist=[] #最后所有比分TFIDF
    idf=[]#idf 值
    idfdir={}
    for line in alllines:#每一个文档
        for wordindex in range(len(allwords)):
                if allwords[wordindex] in line:
                    appearnum[wordindex]+=1#统计总出现,加入列表
    for i in range(len(appearnum)):
        idfnum=appearnum[i]
        num1=math.log(num/idfnum)
        idf.append(num1)#计算
        idfdir[allwords[i]]=num1                              #所有词的字典，idf值
    for li in range(len(alllines)):#遍历所有文档（字典）
        linum=[0]*len(allwords)#ALLWORDS向量
        a=0
        allnum=0
        resultall={}
        for index in range(len(allwords)):#所有的ALLWORDS
            if allwords[index] in alllines[li]:#字典
                tfidfnum=int(alllines[li][allwords[index]])/countnum[li]*idf[index]
                resultall[allwords[index]]=tfidfnum#加入字典TFIDF和字的值
                linum[index] = tfidfnum
                a+=1
                allnum+=tfidfnum
            else:
                linum[index] = 0.0
                resultall[allwords[index]] = 0.0
        resultall = [(k, resultall[k]) for k in sorted(resultall, key=resultall.get, reverse=True)]
        result={}
        for i in range(k):#提取K个有效字
            name1=resultall[i][0]#词
            if resultall[i][1]!=0.0:
                result[name1]=alllines[li][name1]
            else :
                result[name1] = 0
            if name1 not in resultallwords:
                resultallwords.append(name1)
        resultalllines.append(result)#更新所有字典
        print("总共有效个数： ",a,"   总共分数：   ",allnum)
        tfidflist.append(linum)#TFIDF结果，目前没有调用
    # print("结果个数  "+str(len(tfidflist)))
    print("原先总词数：  "+str(len(allwords)))
    print("返回总词数： "+str(len(resultallwords)))
    print("返回后的所有词：   ",resultallwords)
    print("提取 "+str(k)+colored(" 个    使用时间： ","red"), time.clock() - t0)
    return resultalllines,resultallwords,resultcountnum,idfdir#返回所有词典，返回所有词，返回每个文档特征数量,idf值对应词的字典


def createlist(resultalllines,resultallwords):
    t0=time.clock()
    returnlist=[]
    num=len(resultallwords)
    for line in resultalllines:
        linelist=[0]*num
        for i in range(len(resultallwords)):
            if resultallwords[i] in line:
                linelist[i]=line[resultallwords[i]]
        returnlist.append(linelist)
    print(colored("生成总列表时间","red"),time.clock()-t0)
    return returnlist














