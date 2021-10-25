#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用预训练的glove词向量来表征文本
"""
import smart_open
smart_open.open = smart_open.smart_open
import logging
import sys
program = sys.argv[0]
logger = logging.getLogger(program)
from gensim.models import KeyedVectors
import numpy as np
import time
from sklearn.metrics import pairwise_distances
import pandas as pd


def glove2word2vec(glove_vector_file, output_model_file):
    """Convert GloVe vectors into word2vec C format"""
    def get_info(glove_file_name):
        """Return the number of vectors and dimensions in a file in GloVe format."""
        with smart_open.smart_open(glove_file_name) as f:
            num_lines = sum(1 for line in f)
        with smart_open.smart_open(glove_file_name) as f:
            num_dims = len(f.readline().split()) - 1
        return num_lines, num_dims

    def prepend_line(infile, outfile):
        """
        Function to prepend lines using smart_open
        """
        num_lines, dims = get_info(glove_vector_file)
        logger.info('%d lines with %s dimensions' % (num_lines, dims))
        with smart_open.smart_open(infile, 'rb') as old:
            with smart_open.smart_open(outfile, 'wb') as new:
                new.write("{0} {1}\n".format(num_lines, dims).encode('utf-8'))
                for line in old:
                    new.write(line)
        return outfile

    model_file = prepend_line(glove_vector_file, output_model_file)
    logger.info('HGAT %s successfully created !!' % output_model_file)


# 根据预训练模型，生成word2vec向量
class GenerateWordVectors:
    def __init__(self, model_path):
        # 根据传入的预训练模型路径，加载模型
        self.model = KeyedVectors.load_word2vec_format(model_path)

    def embeddings(self, text: list, dimension=200):
        """
        简单的平均加权向量方法，生成句子的表示
        :param text: 包含分词token的list
        :param dimension: 词嵌入维度
        :return: 句子的向量表示
        """
        word_vector = np.zeros(dimension).reshape((1, dimension))
        length = len(text)
        if length == 0:
            length = 1
        for word in text:
            try:
                em = self.model[word]  # 转化为word2vec
                word_vector += em.reshape((1, dimension))
            except KeyError:  # 如果分词没有出现在预训练模型中
                continue  # 相当于加了一个全零向量,类似于unk方法
        word_vector = word_vector.reshape(dimension)
        word_vector /= length  # 简单平均来表示文本
        word_vector[np.isnan(word_vector)] = 100
        return word_vector

    def distance_matrix(self, words, dis=True):
        """
        计算传入文本之间的距离矩阵
        :param corpus: 传入数据集
        :return: 返回距离矩阵
        """
        # 当前传入数据的文本表征
        embeddings = []
        for w in words:
            embeddings.append(self.embeddings(w))
        if dis:
            # 计算欧式距离矩阵
            sim_matrix = pairwise_distances(embeddings, metric='euclidean')
            return np.array(sim_matrix)
        else:
            return np.array(embeddings)
