import time

from algroithms.kernels import distEclud
from numpy import *
from termcolor import *

from process.processmain import processReturn
# kMeans聚类算法
# 基本思想:
# 基于距离和预定义好的类簇数目k,
# 首先,随机选定k个初始类簇中心(不同的类簇中心会导致收敛速度和聚类结果有差别,有可能会陷入局部最优.)
# 其次,计算每个点到每个类簇中心的距离,并将其分配到最近的类簇中
# 第三,重新计算每个类簇的中心
# 第四,重复第二步和第三步直到类簇中心不再发生变化,聚类停止
#
# 二分k-均值聚类算法
# 基本思想
# 该算法首先将所有点作为一个簇,然后将该簇一分为二.之后选择其中一个簇继续进行划分,
# 选择哪一个簇进行划分取决于对"其划分是否可以最大程度降低SSE的值.上述基于SSE的划分过程不断重复,
# 直到得到用户指定的簇数目为止.
#第一个参数，TF-IDF值
#第二个参数 多少个族
#第三个参数
#  距离的函数
#最后结果选用的参数


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  #创建初始中心和误差
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)#创建中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 每个数据点
            minDist = inf
            minIndex = -1
            for j in range(k):#每个族
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True#计算误差最小的确认
            clusterAssment[i, :] = minIndex, minDist ** 2
        # printcentroids
        for cent in range(k):  # recalculate centroidsmean(a,axis=0).tolist()[0]
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean移向平均
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    t0=time.clock()
    dataSet=mat(dataSet)
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))#初始中心点
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # create a list with one centroid
    for j in range(m):  # calc initial Error
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]  # get the data points currently in cluster i
            if len(ptsInCurrCluster)<3:continue
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0], 1])
            # print("选择分离的族", sseSplit,"未分离的族", sseNotSplit,i)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print ('要分离的族:  ', bestCentToSplit)
        print('分离族包含的元素： ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:] = bestClustAss  # reassign new clusters, and SSE
    print(colored(str(k)+"个族生成用时", "red"), time.clock() - t0)
    return centList, clusterAssment.tolist()#返回每个族的空间距离，每个数据所属于的族

def  createPro(centList,clusterAssment,classify,all):#坐标位置，哪个中心，误差率，训练集类别
    t0= time.clock()
    centnum=len(centList)#所有中心数量
    centall=[0 for i in range(centnum)]#初始化所有中心出现的数量
    centclass = [[0 for i in range(all)] for x in range(centnum)]#生成列表[[每一个类别的概率],[],[],[],[]]所有中心
    for i in range(len(clusterAssment)):
        clustindex=int(clusterAssment[i][0]) #哪一个中心
        centclass[clustindex][classify[i]]+=1 #所有族对所有类别的次数
        centall[clustindex]+=1
    finalnum=[]
    for numindex in range(len(centclass)):#计算所有概率
        num=centall[numindex]+1
        pronum=[i/num for i in centclass[numindex]]
        finalnum.append(pronum)
    print(colored("生成所有概率用时", "red"), time.clock() - t0)
    return centList,finalnum#finalnum=[[每一个类别的概率],[],[],[],[]]所有中心

def produceResults(centList,finalnum,testlists):
    returndis=[]#距离
    for testlist in testlists:#每个数据
        dis=[]
        for i in centList:
            dis.append(10/distEclud(mat(testlist),mat(i)))#加入每个数据和每个中心的相对距离
        returndis.append(dis)
    finalnum=mat(finalnum)#生成矩阵
    finalnum=mat(returndis)*finalnum#相乘 生成的数据对每个类型的总距离
    finalnum=finalnum.tolist()
    prediction=[]
    for i in range(len(finalnum)):#选择总体距离最大的       ？？？？？？只选取了最近的一项，没有KNN
        maxnum=-inf
        maxindex=inf
        for num in range(len(finalnum[i])):
            if finalnum[i][num]>maxnum:
                maxnum=finalnum[i][num]
                maxindex=num
        prediction.append(maxindex)
    return prediction

def accuracy(preclassify,classify):
    total=len(classify)
    right=0
    for i in range(total) :
        if preclassify[i]==classify[i]:
            right+=1
            print("第",i,"个正确")
    rightper=right/total
    print(colored("————————————————————————————","red"))
    print(colored("正确率是    ","red"),rightper)
    print(colored("————————————————————————————","red"))
    return rightper


def kMeansmain(dataSet,classify,k,proportion):
    all=len(set(classify))
    print(colored("生成"+str(k)+"个族.......................................................","blue"))
    a=int(len(classify)*proportion)
    trandata=dataSet[:a]
    tranClassify=classify[:a]
    testdata=dataSet[a:]
    testClassify=classify[a:]
    centList, clusterAssment=biKmeans(trandata, k, distMeas=distEclud)#传入数据集，K个中心族，距离公式
    #返回每个族的空间距离，每个数据所属于的族和误差
    centList, finalnum = createPro(centList, clusterAssment,tranClassify,all)#centList中心距离，finalnum=[[每一个类别的概率],[],[],[],[]]所有中心
    prediction = produceResults(centList, finalnum,testdata)
    rightper=accuracy(prediction, testClassify)
    return rightper#正确率


def classifyPro(centList, clusterAssment, classify, all):  # 坐标位置，哪个中心，误差率，训练集类别
    t0 = time.clock()
    centnum = len(centList)  # 所有中心数量
    # centall=[0 for i in range(centnum)]#初始化所有中心出现的数量
    centclass = [[0 for i in range(all)] for i in range(centnum)]  # 生成列表[[每一个类别的概率],[],[],[],[]]所有中心
    for i in range(len(clusterAssment)):
        clustindex = int(clusterAssment[i][0])  # 哪一个中心
        # pro = double(clusterAssment[i][1]) / totalerror
        centclass[clustindex][classify[i]] += 1 # 所有族对所有类别的次数
    finalnum = []
    for j in centclass:
        finalindex = 0
        finalmax=0
        for cent1 in range(len(j)):
            if j[cent1]>finalmax:
                finalindex=cent1
                finalmax=j[cent1]#每个族心中最多的类别
        finalnum.append(finalindex)
    print(colored("确定族属于哪个类", "red"), time.clock() - t0)
    return centList, finalnum  # final每个中心的类别

def belongclassify(testdata,centList, finalnum):
    print(colored(finalnum,"yellow"))
    result=[]
    for data in testdata:
        classifyindex=0
        classifymin = 0
        for i in range(len(centList)):
            distance=distEclud(mat(data),mat(i))#加入每个数据和每个中心的相对距离
            if distance<classifymin:
                classifyindex=finalnum[i]
                classifymin=distance#选择最近的一个点
        result.append(classifyindex)
    return  result
def kMeansmain2(dataSet,classify,k,proportion):
    all=len(set(classify))
    print(colored("生成"+str(k)+"个族.......................................................","blue"))
    a=int(len(classify)*proportion)
    trandata=dataSet[:a]
    tranClassify=classify[:a]
    testdata=dataSet[a:]
    testClassify=classify[a:]
    centList, clusterAssment=biKmeans(trandata, k, distMeas=distEclud)#传入数据集，K个中心族，距离公式
    #返回每个族的空间距离，每个数据所属于的族和误差
    centList, finalnum = classifyPro(centList, clusterAssment,tranClassify,all)#centList中心距离，finalnum=[[每一个类别的概率],[],[],[],[]]所有中心
    prediction = belongclassify(testdata,centList, finalnum)
    rightper=accuracy(prediction, testClassify)
    return rightper#正确率
def itemKMean(dataSet,classify,proportion,kList):
    bestK=0
    bestrightper=0
    for num in kList:
        rightper=kMeansmain(dataSet, classify, num, proportion)
        if bestrightper<rightper:
            bestrightper=rightper
            bestK=num
    print(colored("最佳正确率是： "+ str(bestrightper),"red"))
    print(colored("最佳K个数值是： " + str(bestK), "red"))

def itemKMean2(dataSet,classify,proportion,kList):#按照最近点的分类，有KNN思想
    bestK=0
    bestrightper=0
    for num in kList:
        rightper=kMeansmain2(dataSet, classify, num, proportion)
        if bestrightper<rightper:
            bestrightper=rightper
            bestK=num
    print(colored("最佳正确率是： "+ str(bestrightper),"red"))
    print(colored("最佳K个数值是： " + str(bestK), "red"))
t0=time.clock()
returnlist,classifylist, doctorlist=processReturn(250)
kList=[30]
itemKMean(returnlist,classifylist,0.75,kList)
print(colored("程序总共运行时长", "red"), time.clock() - t0)
