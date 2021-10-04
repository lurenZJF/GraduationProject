#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
"""
常用函数
"""


def build_data(doc):
    """
    根据传入的json数据，构造聚类所用的文本list
    :param doc:
    :return:
    """
    contents = []
    time_info = []
    labels_true = []
    for obj in doc:
        contents.append(obj['text'])  # 构造聚类所需文本信息
        time_info.append(obj['created_at'])
        labels_true.append(obj['event_id'])
    return contents, time_info, labels_true


def distance_analysis(distance, method, index):
    # 绘制图形
    XL = []
    i = 0
    while i < len(distance):
        j = i
        while j < len(distance):
            if distance[i][j] != 3 and distance[i][j] != 0:
                XL.append(distance[i][j])
            j = j + 1
        i = i + 1
    sns.distplot(XL, kde_kws={'bw_adjust': 0.1})
    # plt.show()
    plt.savefig("../Output/"+method+"/"+str(index)+".png", bbox_inches='tight')
    plt.close()