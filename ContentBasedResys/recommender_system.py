import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from IPython import embed

ds = pd.read_csv("sample-data.csv")# ds有两条属性id和description

# Convert a collection of raw documents to a matrix of TF-IDF features
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
# Learn vocabulary and idf, return document-term matrix
tfidf_matrix = tf.fit_transform(ds['description'])

# calcute cosine similarity and its shape is (500, 52262)
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]# 倒序排列，取前99个物品
    # 得到和前99个物品的相似度列表 similar_items的数据类型为list
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]# 除去第0行，因为第0行为待计算的物品本身
    
print('done!')

# 通过item的id得到item的description_name
def item(id):
    # embed()
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary.
def recommend(item_id, num):
    print("推荐与" + item(item_id) + "最相似的前" + str(num) + "个物品")
    print("-------")
    recs = results[item_id][:num]# 取results这个dict的第item_id条数据的前num条“数据”
    for rec in recs:
        print("推荐物品: " + item(rec[1]) + " (相似度得分:" + str(rec[0]) + ")")

recommend(item_id=11, num=5)
