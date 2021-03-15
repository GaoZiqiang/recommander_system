#-*-coding:utf-8-*-
'''
userCF：基于用户的协同过滤算法
'''
from math import sqrt
from IPython import embed
fp = open("uid_score_bid.dat","r")# 使用图书推荐数据集uid_score_bid，数据格式为user_id，用户兴趣度，book_id
 
users = {}# users的数据类型为dict

# 对users和items进行归纳整理
for line in open("uid_score_bid.dat"):
    lines = line.strip().split(",")# 读取一行，得到一条数据记录，下面的示例以第一条数据为例
    # embed()
    if lines[0] not in users:
        users[lines[0]] = {}# 此时的users为{'dingdanglbh': {}}
    users[lines[0]][lines[2]]=float(lines[1])# 此时的users为{'dingdanglbh': {'25862578': 4.0}}，users的类型是一个以user为index的dict
 
# embed()
 
#----------------新增代码段END----------------------
 
 
 
class recommender:
    #data：数据集，这里指users
    #k：表示得出最相近的k的近邻
    #metric：表示使用计算相似度的方法
    #n：表示推荐book的个数
    def __init__(self, data, k=3, metric='pearson', n=12):
 
        self.k = k
        self.n = n
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
 
        self.metric = metric
        if self.metric == 'pearson':
            self.fn = self.pearson# 确认相似度计算函数
        if type(data).__name__ == 'dict':
            self.data = data
      
    def convertProductID2name(self, id):
 
        if id in self.productid2name:
            return self.productid2name[id]
        else:
            return id
 
    #定义的计算相似度的公式，用的是皮尔逊相关系数计算方法
    def pearson(self, rating1, rating2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in rating1:
            if key in rating2:
                n += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        
        #皮尔逊相关系数计算公式 
        denominator = sqrt(sum_x2 - pow(sum_x, 2) / n)  * sqrt(sum_y2 - pow(sum_y, 2) / n)
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator
    # 找到与给定username最近邻的userlist
    def computeNearestNeighbor(self, username):
        distances = []
        # 进行遍历
        for instance in self.data:
            if instance != username:# instance是对应数据记录的username
                distance = self.fn(self.data[username],self.data[instance])# 计算user之间的相似度，distance就是相似度，distance越大表示相似度越高。此处可以参考余弦距离计算法
                distances.append((instance, distance))# 是user之间的相似度
        # 进行排序，找出Nearest Neighbor
        distances.sort(key=lambda artistTuple: artistTuple[1],reverse=True)
        return distances
    
    #推荐算法的主体函数
    def recommend(self, user):
        #定义一个字典，用来存储推荐的书单和分数/权重，因此recommendations的格式为{bookname,score/weight}
        recommendations = {}
        #计算出user与所有其他用户的相似度，返回一个list
        nearest = self.computeNearestNeighbor(user)
        # print nearest
        
	# 用户user对应的books
        userRatings = self.data[user]
	# print userRatings

        totalDistance = 0.0
        # embed()

        #得到最近的k个近邻的总距离 为什么要计算总距离？答：用于下面的归一化
        for i in range(self.k):
            totalDistance += nearest[i][1]
        if totalDistance==0.0:# 说明数据集中只有一个样本
            totalDistance=1.0
            
        #将与user最相近的k个人中 user没有看过的书 推荐给user，并且这里又做了一个分数的计算排名
        for i in range(self.k):
            
            #第i个人的与user的相似度，即为所推荐书的得分/权重weight，转换到[0,1]之间 此为归一化
            weight = nearest[i][1] / totalDistance
            
            #第i个人的name
            name = nearest[i][0]
 
            #第i个用户看过的书和相应的打分
            neighborRatings = self.data[name]
 
            for artist in neighborRatings:# 此处的artist为书名
                if not artist in userRatings:# userRatings为目标user的阅读list
                    if artist not in recommendations:
                        recommendations[artist] = (neighborRatings[artist] * weight)# 将本书的权重weight加入dict
                    else:
                        recommendations[artist] = (recommendations[artist]+ neighborRatings[artist] * weight)
 
        recommendations = list(recommendations.items())
        recommendations = [(self.convertProductID2name(k), v)for (k, v) in recommendations]
        
        #做了一个排序
        recommendations.sort(key=lambda artistTuple: artistTuple[1], reverse = True)
 
        return recommendations[:self.n],nearest
 
def adjustrecommend(id):
    bookid_list = []
    r = recommender(users)
    k,nearuser = r.recommend("%s" % id)
    for i in range(len(k)):
        bookid_list.append(k[i][0])
    return bookid_list,nearuser[:15]#bookid_list推荐书籍的id，nearuser[:15]最近邻的15个用户

if __name__ == '__main__':
    bookid_list,near_list = adjustrecommend("changanamei")
    print ('推荐书的list')
    print (bookid_list)# 推荐书的list
    print ('与指定user最近的前15个用户的排序列表list')
    print (near_list)# 与指定user最近的前15个用户的排序列表list
