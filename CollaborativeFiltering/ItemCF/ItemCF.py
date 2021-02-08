#-*-coding:utf-8-*-
 
'''
ItemCF：基于物品的协同过滤算法
'''
import math
from IPython import embed

class ItemBasedCF:
    def __init__(self,train_file):# 唯一的形参为训练集/数据集的路径
        self.train_file = train_file
        self.readData()
    # 读取数据
    def readData(self):
        # 读取文件，并生成用户-物品（在这里是图书）的评分表和测试集
        self.train = dict()# 生成一个空dict作为用户-物品的评分表，数据类型为dict，dict()函数为python的内置函数
        for line in open(self.train_file):
            # 切割出user,score,item
            user,score,item = line.strip().split(",")
            self.train.setdefault(user,{})# train的key值为user，value值默认为空dict{}
            # embed()
            self.train[user][item] = int(float(score))
        # embed()
 
    def ItemSimilarity(self):
        # 建立物品-物品的共现矩阵
        C = dict()# 物品-物品的共现矩阵，共现矩阵体现了物品之间的相似度
        N = dict()# 物品被多少个不同用户购买
        for user,items in self.train.items():
            for i in items.keys():# 此处的i其实是item的名字（本数据集中是书名）
                N.setdefault(i,0)
                N[i] += 1# 表示这本书被user这个用户购买过，也表示对物品i感兴趣的用户数量加1
                C.setdefault(i,{})
                for j in items.keys():# 再遍历一次该user对应的items
                    if i == j : continue
                    C[i].setdefault(j,0)# 设置二级dict
                    C[i][j] += 1
                # embed()# 最终得到物品-物品的共现矩阵
        # 计算相似度矩阵
        self.W = dict()
        for i,related_items in C.items():
            self.W.setdefault(i,{})
            for j,cij in related_items.items():
                self.W[i][j] = cij / (math.sqrt(N[i] * N[j]))# 计算喜欢物品i的用户中同时对物品j的感兴趣的可能性，在这里被叫做两种物品之间的相似度
        # embed()
        return self.W
 
    # 给用户user推荐，前K个相关物品，即最相似的K个物品
    def Recommend(self,user,K=3,N=10):
        rank = dict()
        action_item = self.train[user]# 用户user产生过行为的item和评分，就是用户喜欢的物品的集合N(u)
        for item,score in action_item.items():
            for j,wj in sorted(self.W[item].items(),key=lambda x:x[1],reverse=True)[0:K]:# 找出与集合N(u)中的item的相似度排名前三物品的列表
                if j in action_item.keys():# 如果得到的j是原集合N(u)中已存在的item，则忽略
                    continue
                rank.setdefault(j,0)
                rank[j] += score * wj# 这里的score就是rui
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:N])

if __name__ == "__main__":
    # 声明一个ItemBased推荐的对象    
    Item = ItemBasedCF("uid_score_bid.dat")
    # embed()
    Item.ItemSimilarity()
    recommendDic = Item.Recommend("ruisuyun630")
    # embed()
    print("------为用户ruisuyun630推荐的物品列表------")
    for k,v in recommendDic.items():
        print ("推荐item:",k,"\t","兴趣度:",v)
