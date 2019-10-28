# 导入库
import numpy as np
import pandas as pd
import jieba
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 导入文本，停用词表
data = pd.read_excel(r'C:\Users\Administrator\Desktop\data.xlsx',sheetname = 1)
stop_words = list(pd.read_csv(r'C:\Users\Administrator\Desktop\停用词表.txt',
	names = ['word'],sep = 'aaa',encoding = 'UTF-8').word)

# 分词
data_part_sk = [' '.join([w for w in jieba.lcut(w) if w not in stop_words and len(w) > 1]) for w in data.text]
data_part_gs = [[w for w in jieba.lcut(w) if w not in stop_words and len(w) > 1] for w in data.text]

# 构建TFIDF
countvec = CountVectorizer(min_df = 3,ngram_range = (1,1))
x = countvec.fit_transform(data_part_sk)
tfidf = TfidfTransformer(norm = 'l1')
x_tf = tfidf.fit_transform(x)

# 构建Word2Vec
w2vmodel = Word2Vec(sg = 0,window = 5,alpha = 0.025,size = 50,min_count = 2,iter = 5)
w2vmodel.build_vocab(data_part_gs)
w2vmodel.train(data_part_gs,total_examples = w2vmodel.corpus_count,epochs = 10)

# TFIDF考虑Word2Vec平均(最终结果以TFIDF重点词为基准的向量)
# vector = (Σwi*ti)/tn
size = w2vmodel.layer1_size
vec = np.zeros(shape=(1,size), dtype=np.float32)
docvec = []
word = countvec.get_feature_names()
weight = x_tf.toarray()
length = len(word)
for i in range(len(weight)):
    for j in range(len(word)):
        try:
            vec_word_weight = w2vmodel.wv[word[j]] * weight[i][j]
            vec += vec_word_weight
        except:
            length -= 1
            continue
        vec = vec / length
    docvec.append(vec)

# Word2Vec平均考虑TFIDF加权(最终结果为Word2Vec维度向量)
# vector = (Σwi*ti)/length
size = w2vmodel.layer1_size
vec = np.zeros(shape=(1,size), dtype=np.float32)
docvec = []
word = countvec.get_feature_names()
weight = x_tf.toarray()

for i in range(len(weight)):
    length = len(data_part_gs[i])
    for w in data_part_gs[i]:
        try:
            vec_word_weight = w2vmodel.wv[w] * weight[i][word.index(w)]
            vec += vec_word_weight
        except:
            length -= 1
            continue
        vec = vec / length
    docvec.append(vec)