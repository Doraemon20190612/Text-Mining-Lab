# 导入库
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer

# 导入文本，停用词表
data = pd.read_excel(r'C:\Users\Administrator\Desktop\data.xlsx',sheetname = 1)
stop_words = list(pd.read_csv(r'C:\Users\Administrator\Desktop\停用词表.txt',
	names = ['word'],sep = 'aaa',encoding = 'UTF-8').word)

# 分词
data_part = [' '.join([w for w in jieba.lcut(w) if w not in stop_words and len(w) > 1]) for w in data.text]

# Count Vector
countvec = CountVectorizer(min_df = 3,ngram_range = (1,1))
x = countvec.fit_transform(data_part)

# 输出
print(countvec)
print(x)
print(countvec.get_feature_names())
print(countvec.vocabulary_)
print(x.toarray())