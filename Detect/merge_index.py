#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
在固定文本表征的前提下，实现流式聚类
"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.stream import MONGO
from Baseline.tf_idf import *
from TextFiltering.twitter_preprocessor import TwitterPreprocessor
from Baseline.cluster_function import *
# 初始化分词实例
Cut = TwitterPreprocessor()


def single_cluster(doc, eps_value, sample_value):
    # 抽取单词和实体
    token_w = []
    token_e = []
    for c in doc:
        words = Cut.get_token(c["text"])
        entity = Cut.entity_recognition(c["text"])
        token_w.append(' '.join(words))
        token_e.append(' '.join(entity))
    w_embeddings = feature_vector(token_w)
    e_embeddings = feature_vector(token_e)
    distance = w_embeddings + 2 * e_embeddings
    # print("distance:", str(len(distance)))
    db = my_db(eps=eps_value, min_sample=sample_value, metric='precomputed', corpus_distance=distance)
    # 需要返回数据集和数据集对应的“表征点数量”
    cluster, cluster_point, noise = presentation_point(db, doc)
    return cluster, cluster_point, noise


def our_method(res, eps=1.7,  second_eps=1.7, const_sample=3):
    """
    传入的数据是Json包类型
    :param res:
    :return:
    """
    # 对原始推文第一次聚类
    clusters, core, noise_data = single_cluster(res, eps_value=eps, sample_value=const_sample)
    # 判断已经聚类得到的数据否需要进行deep clustering
    event_queue = []
    result = []
    for i in range(len(clusters)):
        if len(core[i]) > 15:  # 如果缩减后的核心点数量仍然很多
            event_queue.append(clusters[i])
        else:
            result.append(clusters[i])
    while event_queue:
        # print(len(event_queue), flush=True)
        new_clusters, new_core, new_noise = single_cluster(event_queue[0], eps_value=second_eps, sample_value=const_sample)
        noise_data.extend(new_noise)
        for j in range(len(new_clusters)):
            if len(new_core[j]) > 15:  # 如果缩减后的核心点数量仍然很多
                # print(len(new_clusters[j]), flush=True)
                event_queue.append(new_clusters[j])
            else:
                result.append(new_clusters[j])
        event_queue.pop(0)
        if second_eps > 0.5:
            second_eps -= 0.2
            const_sample += 2
    # 对待每个event_cluster执行Event_merge









if __name__ == "__main__":
    pass

