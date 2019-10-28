# 导入库
import numpy as np
import pandas as pd
import jieba
import gensim
from gensim.models import doc2vec

# 导入文本，停用词表
data = pd.read_excel(r'C:\Users\Administrator\Desktop\data.xlsx',sheetname = 1)
stop_words = list(pd.read_csv(r'C:\Users\Administrator\Desktop\停用词表.txt',
	names = ['word'],sep = 'aaa',encoding = 'UTF-8').word)

# 自定义分词函数
def m_cut(intxt):
    return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) > 1]

# 自定义TaggedDocument生成函数
def m_doc(doclist):
    reslist = []
    for i,doc in enumerate(doclist):
        reslist.append(doc2vec.TaggedDocument(m_cut(doc),[i]))
    return reslist

# 生成TaggedDocument
data_corp = m_doc(data.text)

# doc2vec模型拟合
d2vmodel = gensim.models.Doc2Vec(vector_size = 50,window = 20,min_count = 2,iter = 5)
d2vmodel.build_vocab(data_corp)

# 输出
for i in range(len(data.text)):
    print(d2vmodel.docvecs[i])