#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import datetime
import logging
import pandas as pd
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.stream import MONGO
from Baseline.tf_idf import *
from Detect.utils import *
from Baseline.cluster_function import *
from Baseline.eventx import *
# 日志信息
log_console = logging.StreamHandler(sys.stdout)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.INFO)
default_logger.addHandler(log_console)
# 初始化分词实例
Cut = TwitterPreprocessor()

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


def stream_supervised_cluster(method = "TF_IDF"):
    """
    衡量模型的效果
    :param method:
    :return:
    """
    # 生成时间信息
    time_list = range_date("2012-10-10", "2012-11-07")
    N = len(time_list)
    # 调用数据查询方法
    MG = MONGO("TwitterEvent2012", "tweets")
    # 存储每次的结果
    result = []
    for i in range(N - 1):
        # 根据时间信息进行信息检索；返回的是游标，会随着遍历而移动
        res = MG.query(time_list[i], time_list[i + 1])
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
            distance_analysis(distance, method, i)
            db = my_db(eps=2.8, min_sample=3, metric='precomputed', corpus_distance=distance)
            # 观察聚类结果
            ans = supervised_show(labels_true, db)
        elif method == "EVENTX":
            ans = event_method(res)
        elif method == "LDA":
            ans = 0
        elif method == "GLOVE":
            ans = 0
        elif method == "DEEP":
            ans = 0
        else:
            ans = 0
            pass
        result.append([time_list[i]]+ans)
    if method in ["TF_IDF", "DEEP"]:
        result = pd.DataFrame(result, columns=['time', "NMI", "ARS", "event_number", "cluster_number", "noise_number"])
    else:
        result = pd.DataFrame(result, columns=['time', "NMI", "ARS"])
    result.to_csv('../Output/' + method + "/metric.csv", index=False)


if __name__ == "__main__":
    stream_supervised_cluster(method="EVENTX")
    # stream_supervised_cluster()
