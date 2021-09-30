#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
聚类相关函数
"""

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/dell/GraduationProject/')
from Baseline.core_dbscan import DBSCAN


def my_db(eps, min_sample, metric, corpus_distance):
    """
    实现单次聚类
    :param eps: 邻ϵ-域的距离阈值，和样本距离超过ϵ的样本点不在ϵ-邻域内；
    eps过大，更多的点会落在核心对象的邻域内，此时类别数会减少；反之类别数增大；
    :param min_sample: 样本点要成为核心对象所需要的ϵ-邻域的样本数阈值；通常和eps一起调参；
    在eps一定的情况下，min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多；
    反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少
    :param metric:
    :param corpus_distance: 根据数据集计算得到的距离矩阵
    :return: 聚类后的标签信息，核心点信息
    """
    clf = DBSCAN(eps=eps, min_samples=min_sample, metric=metric)
    # 根据数据训练模型
    db = clf.fit(corpus_distance)
    return db


def presentation_point(db, corpus):
    """
    计算核心点
    :param db: 训练得到模型
    :param corpus: 原始文本数据
    :return: 事件簇列表，事件簇核心点列表, 噪音数据列表
    """
    # 将dataframe改成numpy
    corpus = np.array(corpus)
    labels = db.labels_  # 获取预测标签数据
    # 获取聚类数量
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # 生成存放聚类数据的容器[[],[],[],[],[]...]
    clustered_sentences = [[] for i in range(n_clusters_)]
    # 生成存放核心点数据索引的容器[[],[],[],...,[]]
    clustered_core_index = [[] for i in range(n_clusters_)]
    # 存放DBSCAN聚类中不符合聚类条件的数据;
    noise_data = []
    # 将同一类文本放到一个list中
    for sentence_id, cluster_id in enumerate(labels):
        if cluster_id > -1:  # 过滤噪音点
            clustered_sentences[cluster_id].append(corpus[sentence_id])
            if sentence_id in db.core_sample_indices_:
                clustered_core_index[cluster_id].append(sentence_id)
        else:
            noise_data.append(corpus[sentence_id])
    # 从核心点中根据给定条件缩小核心点的数量
    reduce_clustered_core = [[] for i in range(n_clusters_)]
    for i in range(n_clusters_):
        core = clustered_core_index[i]
        # 标记是否被访问过的数组
        visited_core = [-1 for i in range(len(core))]
        # 开始核心点信息
        j = 0
        while j < len(core):  # 第一次遍历
            if visited_core[j] == -1:  # 如果当前的点不在某个核心点的邻域内
                k = j + 1
                while k < len(core):  # 第二次遍历
                    # 计算这个点，是否在已经确定的“代表”点邻域内\
                    if visited_core[k] == -1:
                        if core[k] in db.neighborhoods[core[j]]:
                            # 如果当前的点在 core[j]的邻域内，则表示为访问过
                            visited_core[k] = 1
                    k = k + 1
            j = j + 1
        # 取出所有标记为-1的点，这些即为最终的代表点
        j = 0
        while j < len(core):
            if visited_core[j] == -1:
                reduce_clustered_core[i].append(corpus[core[j]])
            j = j + 1

    return clustered_sentences, reduce_clustered_core, noise_data


def supervised_show(corpus, db):
    """
    评估数据效果
    :param corpus: 数据集
    :param db: 聚类结果
    :return: NMI值和V-measure值
    """
    labels = db.labels_  # 获取预测标签数据
    # 获取聚类数量
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # 获取噪音点信息，在DBSCAN聚类中，噪音点用-1标记
    n_noise_ = list(labels).count(-1)
    # 输出聚类结果
    print('Estimated number of clusters: %d' % n_clusters_, flush=True)
    print('Estimated number of noise points: %d' % n_noise_, flush=True)
    # clustered_sentences, reduce_clustered_core, noise_data = presentation_point(db, corpus)
    # print("the number of events: ", len(corpus["event_id"].unique()))

    # 输出过滤噪音后的聚类结果
    # for i, cluster in enumerate(clustered_sentences):
    #     print("Cluster ", i + 1, flush=True)
    #     # print(cluster)
    #     # print('Reduce Cluster core:')
    #     # print(reduce_clustered_core[i])
    #     print('簇文本数目:', len(cluster),
    #           '簇代表核心文本数目:', len(reduce_clustered_core[i]), flush=True)
    # 聚类指标信息
    labels_true = corpus["event_id"]
    NMI = metrics.normalized_mutual_info_score(labels_true,labels)
    NMI = round(NMI, 3)
    ars = metrics.adjusted_rand_score(labels_true, labels)
    ars = round(ars, 3)
    return [NMI, ars, len(corpus["event_id"].unique()), n_clusters_, n_noise_]


def our_metric(labels_true, labels):
    """

    :param labels_true:
    :param labels:
    :return:
    """
    # 获取聚类数量
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # 获取噪音点信息，在DBSCAN聚类中，噪音点用-1标记
    n_noise_ = list(labels).count(-1)
    # 输出聚类结果
    # print('Estimated number of clusters: %d' % n_clusters_, flush=True)
    # print('Estimated number of noise points: %d' % n_noise_, flush=True)
    # print("The number of events：", len(labels_true.unique()))
    NMI = metrics.normalized_mutual_info_score(labels_true, labels)
    NMI = round(NMI, 3)
    ars = metrics.adjusted_rand_score(labels_true, labels)
    ars = round(ars, 3)
    return [NMI, ars, len(labels_true.unique()), n_clusters_, n_noise_]
