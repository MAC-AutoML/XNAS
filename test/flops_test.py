import numpy as np


def getw(w):
    return int(w*100)


def get_path(n, weight, constraint, FLOPS):
    n=20
    weight=[np.random.rand(8) for i in range(n)]
    constraint=[np.random.rand(8) for i in range(n)]
    print(weight)
    print(constraint)
    FLOPS=10

    max_weight=0
    for i in range(n):
        max_weight+=int(np.max(weight[i])*100)  #保留小数点后两位
    print('max_weight:', max_weight)      #weigth最大可能取的值，作为状态的上限

    dp=[[FLOPS*10 for i in range(max_weight+5)] for i in range(n)]
    #定义dp[n][max_weight] dp[i][j]表示第i个节点，在权值和为j的情况下，能取到的最小的限制值
    pre=np.zeros((n,max_weight+5), int)   #记录pre[i][j]的前驱节点，是由上个节点哪个状态(j)转移到的
    chose=np.zeros((n, max_weight+5), int)   #记录dp[i][j]这个状态在节点i是选择了哪条路径

    for i in range(8):  #对第0个节点初始化
        w=getw(weight[0][i])
        dp[0][w]=constraint[0][i]
        pre[0][w]=-1     #-1表示没有前驱
        chose[0][w]=i
    ans=0   #记录限制下可以取得的最大权值
    endk=0  #记录最后一个节点取了哪条path
    for i in range(1, n):  #遍历每个分组
        for j in range(max_weight+1):   #遍历每个容量，即权值
            for k in range(8):    #遍历当前分组种可选的物品,即路径
                w=getw(weight[i][k])
                if(j>=w):
                    if(dp[i][j]>dp[i-1][j-w]+constraint[i][k]):
                        dp[i][j]=dp[i-1][j-w]+constraint[i][k]
                        pre[i][j]=j-w  #dp[i][j]由dp[i-1][j-w]转移过来，所以前驱是(i-1,j-w)
                        chose[i][j]=k  #选择了第k个物品
                if i==n-1:
                    if dp[i][j]<=FLOPS and j>ans:
                        ans=j
                        endk=k
    path=[]
    path.append(endk)
    nowj=ans
    nownode=n-1
    while(pre[nownode][nowj]!=-1):  #根据记录的前驱不断回溯找到每个分组选择了哪个物品，即每个节点选择了哪条边
        nowj=pre[nownode][nowj]
        nownode-=1
        print(dp[nownode][nowj])
        print(chose[nownode][nowj])
        path.append(chose[nownode][nowj])

    path=path[::-1]
    print('max_weight:',ans)
    print('path:',path)

    return path


def main():
    get_path(0, 0, 0, 0)


if __name__=='__main__':
    main()