# 导入库
import numpy as np
import pandas as pd
import jieba
from gensim.models.word2vec import Word2Vec

# 导入文本，停用词表
data = pd.read_excel(r'C:\Users\Administrator\Desktop\data.xlsx',sheetname = 1)
stop_words = list(pd.read_csv(r'C:\Users\Administrator\Desktop\停用词表.txt',
	names = ['word'],sep = 'aaa',encoding = 'UTF-8').word)

# 分词
data_part = [[w for w in jieba.lcut(w) if w not in stop_words and len(w) > 1] for w in data.text]

# 设定神经网络参数
n_dim = 50
w2vmodel = Word2Vec(sg = 0,window = 5,alpha = 0.025,size = n_dim,min_count = 2,iter = 5)
w2vmodel.build_vocab(data_part)
# 训练网络
w2vmodel.train(data_part,total_examples = w2vmodel.corpus_count,epochs = 10)

w2vmodel.wv.vocab # 显示词表
w2vmodel.wv["美国"] # 显示词向量

# 将词向量转化为文本向量（词向量平均法）
size = w2vmodel.layer1_size # 定义向量维度尺寸
vec = np.zeros(shape=(1, size), dtype=np.float32) # 定义初始化向量（全为0）
docvec = [] # 定义文本向量空集
for i in range(len(data_part)): # 循环文本数量
    length = len(data_part[i]) # 定义每篇文本的词数量
    for word in data_part[i]: # 循环每篇文本的词
        try:
            vec += w2vmodel.wv[word] # 词向量求和
        except:
            length -= 1 # 词数量确定
            continue
        vec = vec / length # 词向量平均
    docvec.append(vec) # 平均后的词向量添加至空集