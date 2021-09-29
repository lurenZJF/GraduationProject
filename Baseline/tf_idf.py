#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
func: 用TF_IDF向量表示文本
"""
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import linear_kernel


def feature_vector(sentences: list, matrix=True):
    """
    用TF-IDF来表征数据
    :param sentences: 文本分词后的形式
    :param matrix: 是否返回相似度矩阵
    :return:
    """
    # 词频矩阵 Frequency Matrix Of Words
    # sublinear_tf,是否应用子线性tf缩放，即用1 + log（tf）替换tf；
    # max_df：float in range [ 0.0，1.0 ]或int，default = 1.0；当构建词汇时，忽略文档频率严格高于给定阈值（语料库特定停止词）的术语。
    vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    transformer = TfidfTransformer()
    # Fit Raw Documents
    freq_words_matrix = vertorizer.fit_transform(sentences)
    # Get Words Of Bag
    words = vertorizer.get_feature_names()
    # 文本中各个词的TFIDF值
    """
    (0, 1006)	0.8289825648683835
    (0, 1459)	0.5592744470689112
    (1, 3102)	0.12414810923229627
    (1, 2370)	0.2765833720261032
    ...
    """
    tfidf = transformer.fit_transform(freq_words_matrix)
    # w[i][j] represents word j's weight in text class i
    weight = freq_words_matrix.toarray()  # 计算相似度或者PCA时使用
    # print(weight)
    if matrix:
        S = len(sentences)
        i = 0
        """
        # 计算欧式距离代码
        distance_matrix = np.zeros((S, S))  # 创建一个距离矩阵
        while i < S:
            j = i
            while j < S:
                d = euclidean_distances([weight[i]], [weight[j]])
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d
                j = j+1
            i = i+1
        return distance_matrix
        """
        # 计算余弦距离
        sim_matrix = []
        while i < S:
            cosine_similarities = linear_kernel(tfidf[i], tfidf).flatten()
            cosine_similarities = 1 - cosine_similarities
            cosine = []
            for c in cosine_similarities:
                c = int(c*1000)/1000
                cosine.append(c)
            sim_matrix.append(cosine)
            i = i+1
        return np.array(sim_matrix)
    return weight
