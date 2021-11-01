#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Bert模型来实现文本表征
在开始执行程序前需要先启动服务
bert-serving-start -model_dir /tmp/cased_L-12_H-768_A-12/ -num_worker=4
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from bert_serving.client import BertClient
bc = BertClient()


def bert_embedding(sentence: list, dis=True):
    """
    调用bert as service 对文本进行编码
    :param sentence: 文本，list格式
    :param dis: 是否求距离矩阵
    :return:
    """
    embedding = bc.encode(sentence)
    if dis:
        # 计算欧式距离矩阵
        sim_matrix = pairwise_distances(embedding, metric='euclidean')
        return np.array(sim_matrix)
    else:
        return np.array(embedding)

