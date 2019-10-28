# 导入相关库
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer

# 导入文本，停用词表
data = pd.read_excel(r'C:\Users\Administrator\Desktop\data.xlsx',sheetname = 1)
stop_words = list(pd.read_csv(r'C:\Users\Administrator\Desktop\停用词表.txt',
	names = ['word'],sep = 'aaa',encoding = 'UTF-8').word)

# 分词
data_part = [' '.join([w for w in jieba.cut(w) if w not in stop_words and len(w) > 1]) for w in data.text]

# 第一层文本向量转换（Count Vector）
countvec = CountVectorizer(min_df = 5,ngram_range = (1,1))
x = countvec.fit_transform(data_part)

# 输出
print(countvec)
print(x)
print(countvec.get_feature_names())
print(countvec.vocabulary_)
print(x.toarray())

# 第二层文本向量转换（Co-Occurence Vector）
xc = (x.T * x)
xc_df = pd.DataFrame(xc.todense(), columns=countvec.get_feature_names(), index=countvec.get_feature_names())

# 输出
print(xc)
print(xc.toarray())
print(xc_df)
matrix = xc.toarray()

# 应用
# 奇异值分解
U, s, V = np.linalg.svd(matrix)

# 聚类
X = -U[:,0:2]
from sklearn.cluster import KMeans
labels = KMeans(n_clusters=2).fit(X).labels_
colors = ('white','green')

# 可视化
from matplotlib import pyplot as plt
plt.figure(figsize=(10,10))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

for word in countvec.vocabulary_.keys():
    i = countvec.vocabulary_[word]
    plt.scatter(X[i, 1], X[i, 0], c=colors[labels[i]], s=400, alpha=0.4)
    plt.text(X[i, 1], X[i, 0], word, ha='center', va='center')

plt.show()