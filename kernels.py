from numpy import *
def distEclud(v1,v2):
    return sqrt(sum(power(v1-v2,2)))


def cossim(a,b):
    num = len(a)
    newp1 = []
    newp2 = []
    for i in range(num):
        if (a[i] != 0 or a[i] != 0):
            newp1.append(a[i])
            newp2.append(b[i])
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(newp1, newp2):
        part_up += a1 * b1
        a_sq += a1 ** 2
        b_sq += b1 ** 2
    part_down = math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return 0
    else:
        return part_up / part_down
 # 返回两个人的皮尔逊相关系数
def sim_pearson( p1, p2):

    num=len(p1)
    newp1=[]
    newp2=[]
    for i in range(num):
        if (p1[i]!=0 or p2[i]!=0):
            newp1.append(p1[i])
            newp2.append(p2[i])

    # 如果两者没有共同之处，则返回0
    if len(newp2)== 0: return 0
    # 对共同拥有的物品的评分，分别求和
    sum1 = sum(newp1)
    sum2 = sum(newp2)
    # 求平方和
    sum1Sq = sum([pow(it, 2) for it in newp1])
    sum2Sq = sum([pow(it, 2) for it in newp2])
    # 求乘积之和
    pSum = sum([newp1[it] * newp2[it] for it in range(len(newp1))])
    # 计算皮尔逊评价值
    n=len(newp1)
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1,2) / n) * (sum2Sq - pow(sum2,2) / n))

    if den == 0: return 0

    r =abs( num / den)

    return r

