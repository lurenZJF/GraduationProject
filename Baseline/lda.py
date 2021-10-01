#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
训练LDA模型
"""
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import pairwise_distances


def run_lda(words):
    """
    训练LDA模型，并用话题表征文本
    :param words: 词汇列表
    :return: 欧式距离矩阵
    """
    # 根据文本信息创建词典
    common_dictionary = Dictionary(words)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    common_corpus = [common_dictionary.doc2bow(text) for text in words]
    # Train the model on the corpus.
    lda = LdaModel(common_corpus, num_topics=40, alpha='auto', eval_every=5)
    # 主题矩阵
    ldainfer = lda.inference(common_corpus)[0]
    # 计算距离
    # 计算欧式距离矩阵
    X = pairwise_distances(ldainfer, metric='euclidean')
    return X