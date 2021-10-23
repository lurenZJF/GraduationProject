#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import datetime
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.stream import MONGO
from Baseline.tf_idf import *
from Detect.utils import *
from Baseline.cluster_function import *
from Baseline.eventx import *
from Baseline.lda import *
from Baseline.glove2vec import *
# from TextFiltering.SHIN import *
from TextFiltering.SHAN import *
# 初始化分词实例
Cut = TwitterPreprocessor()
G = GenerateWordVectors("../Static/glove2word2vec.txt")

"""
func: 流式聚类主体函数
"""


def range_date(start, end, step=1, format="%Y-%m-%d"):
    """
    生成查询时间列表
    :param start: 起始时间
    :param end: 结束时间
    :param step: 步长
    :param format: 时间格式
    :return: 查询时间列表
    """
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]


def deep_cluster(doc, eps, sample, second_eps):
    """
    深度聚类
    :param doc:
    :param eps:
    :param sample:
    :param second_eps:
    :return:
    """
    # 将数据转化为json
    corpus = []
    labels_true = []
    print("deep cluster", flush=True)
    for d in doc:
        corpus.append(d)
        labels_true.append(d["event_id"])
    # print(corpus, flush=True)
    print("传入数据大小：", str(len(corpus)))

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

    # 对原始推文第一次聚类
    clusters, core, noise_data = single_cluster(corpus, eps_value=eps, sample_value=sample)
    # 判断已经聚类得到的数据否需要进行deep clustering
    event_queue = []
    result = []
    for i in range(len(clusters)):
        if len(core[i]) > 15:  # 如果缩减后的核心点数量仍然很多
            event_queue.append(clusters[i])
        else:
            result.append(clusters[i])
    # print("第一次聚类结果：")
    # print("result:", str(len(result)), "待聚类: ", str(len(event_queue)),flush=True)
    while event_queue:
        # print(len(event_queue), flush=True)
        new_clusters, new_core, new_noise = single_cluster(event_queue[0], eps_value=second_eps, sample_value=sample)
        # print(len(new_clusters), flush=True)
        noise_data.extend(new_noise)
        for j in range(len(new_clusters)):
            if len(new_core[j]) > 15:  # 如果缩减后的核心点数量仍然很多
                # print(len(new_clusters[j]), flush=True)
                event_queue.append(new_clusters[j])
            else:
                result.append(new_clusters[j])
        # print("****")
        # 删除队列元素
        event_queue.pop(0)
        if second_eps > 0.5:
            second_eps -= 0.2
            sample += 2
    # 返回最终结果
    ans_labels = []
    for c in corpus:
        ans_labels.append([c["_id"], -1])
    ans_labels = pd.DataFrame(data=ans_labels, columns=['tweet_id', "labels"])
    for i in range(len(result)):
        c = result[i]
        for j in range(len(c)):
            ans_labels.loc[ans_labels.tweet_id == c[j]["_id"], 'labels'] = i
    # 计算各项指标
    NMI = metrics.normalized_mutual_info_score(labels_true, ans_labels["labels"])
    NMI = round(NMI, 3)
    ars = metrics.adjusted_rand_score(labels_true, ans_labels["labels"])
    ars = round(ars, 3)
    return [NMI, ars]


def stream_supervised_cluster(method = "TF_IDF"):
    """
    衡量模型的效果
    :param method:
    :return:
    """
    # 生成时间信息
    time_list = range_date("2012-10-10", "2012-11-07")
    # time_list = range_date("2012-10-16", "2012-10-18")
    N = len(time_list)
    # 调用数据查询方法
    MG = MONGO("TwitterEvent2012", "tweets")
    # 存储每次的结果
    result = []
    # 开始程序计时
    start_time = time.time()
    for i in range(N - 1):
        # 根据时间信息进行信息检索；返回的是游标，会随着遍历而移动
        res = MG.query(time_list[i], time_list[i + 1])
        # res = MG.query(time_list[0], time_list[-1])
        if method == "TF_IDF":
            # 解析数据
            contents, time_info, labels_true = build_data(res)
            # 抽取单词和实体
            token_w = []
            token_e = []
            for c in contents:
                words = Cut.get_token(c)
                entity = Cut.entity_recognition(c)
                token_w.append(' '.join(words))
                token_e.append(' '.join(entity))
            w_embeddings = feature_vector(token_w)
            e_embeddings = feature_vector(token_e)
            distance = w_embeddings + 2 * e_embeddings
            # distance_analysis(distance, method, i)
            db = my_db(eps=1.7, min_sample=5, metric='precomputed', corpus_distance=distance)
            # 观察聚类结果
            ans = supervised_show(labels_true, db)
        elif method == "EVENTX":
            ans = event_method(res)
        elif method == "LDA":
            contents, time_info, labels_true = build_data(res)
            token_w = []
            for c in contents:
                words = Cut.get_token(c)
                token_w.append(words)
            distance = run_lda(token_w)
            # distance_analysis(distance, method, i)
            db = my_db(eps=2, min_sample=2, metric='precomputed', corpus_distance=distance)
            ans = supervised_show(labels_true, db)
        elif method == "GLOVE":
            contents, time_info, labels_true = build_data(res)
            token_w = []
            for c in contents:
                words = Cut.get_token(c)
                token_w.append(words)
            distance = G.distance_matrix(token_w)
            # distance_analysis(distance, method, i)
            db = my_db(eps=1.8, min_sample=3, metric='precomputed', corpus_distance=distance)
            ans = supervised_show(labels_true, db)
        elif method == "DEEP":
            ans = deep_cluster(res, 1.7, 5, 1.7)
        else:
            # distance, labels_true = stream_information(res)
            # distance_analysis(distance, method, i)
            # np.savez("../Output/HAN_distance"+str(i)+".npz", distance=np.array(distance))
            distance = np.load("../Output/HAN_distance"+str(i)+".npz")
            # 聚类
            # 解析数据
            contents, time_info, labels_true = build_data(res)
            db = my_db(eps=3, min_sample=3, metric='precomputed', corpus_distance=distance["distance"])
            ans = supervised_show(labels_true, db)
            print(ans)
        result.append([time_list[i]]+ans)
    end_time = time.time()
    print("cost time:", end_time-start_time, flush=True)
    # exit()
    if method in ["TF_IDF", "LDA", "GLOVE", "HAN"]:
        result = pd.DataFrame(result, columns=['time', "NMI", "ARS", "event_number", "cluster_number", "noise_number"])
    else:
        result = pd.DataFrame(result, columns=['time', "NMI", "ARS"])
    result.to_csv('../Output/' + method + "/metric.csv", index=False)


if __name__ == "__main__":
    # stream_supervised_cluster()
    # stream_supervised_cluster(method="EVENTX")
    # stream_supervised_cluster(method="LDA")
    # stream_supervised_cluster(method="DEEP")
    stream_supervised_cluster(method="HAN")
