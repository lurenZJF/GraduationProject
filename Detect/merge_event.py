#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
实现事件时序合并算法
"""
import numpy as np
from collections import Counter
from summa import keywords
from nltk.corpus import stopwords


def word2tf(word_list1, word_list2):
    """
    词袋模型
    :param word_list1:
    :param word_list2:
    :return:
    """
    # 转成counter不需要考虑0的情况
    words1_dict = Counter(word_list1)
    words2_dict = Counter(word_list2)
    bags = set(words1_dict.keys()).union(set(words2_dict.keys()))
    # 转成list对debug比较方便吗，防止循环集合每次结果不一致
    bags = sorted(list(bags))
    vec_words1 = [words1_dict[i] for i in bags]
    vec_words2 = [words2_dict[i] for i in bags]
    # 转numpy
    vec_words1 = np.asarray(vec_words1, dtype=np.float)
    vec_words2 = np.asarray(vec_words2, dtype=np.float)
    return vec_words1, vec_words2


def cosine_similarity(v1, v2):
    """
    计算余弦相似度
    :param v1:
    :param v2:
    :return:
    """
    # 余弦相似度
    v1, v2 = np.asarray(v1, dtype=np.float), np.asarray(v2, dtype=np.float)
    up = np.dot(v1, v2)
    down = np.linalg.norm(v1) * np.linalg.norm(v2)
    return round(up / down, 3)


def get_keywords(corpus, flag=False):
    """
    根据传入的数据提取关键词
    :param corpus: DataFrame格式
    :param flag: 提取关键词的方法
    :return:
    """
    # 根据事件数量提取关键词
    k_num = corpus.shape[0] // 10
    if k_num < 5:
        k_num = 5
    if k_num > 10:
        k_num = 10
    if flag:
        # 构造成句子格式
        sentence = ""
        for s in corpus["text"]:
            sentence += s
        # print(sentence)
        # print(len(sentence))
        # 提取关键词
        key = keywords.keywords(sentence, ratio=0.8, split=True, additional_stopwords=stopwords.words('english'))
    else:
        # 采用词频统计法
        en = []
        for e in corpus["entity"]:
            en += e
        collection_words = Counter(en)
        key = []
        for k in collection_words.most_common(k_num):
            key.append(k[0].lower())
    if len(key) > k_num:
        key = key[:k_num]
    return key


def event_merge(event):
    """
    在给定时间范围内的事件簇中，进行事件合并
    :param event: 事件
    :return:
    """
    key_words = event['keywords']
    # 如果当前数据库为空





