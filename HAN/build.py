#!/usr/bin/env python
# -*- coding:utf-8 -*-
import dgl
import scipy.sparse as sp
import pandas as pd


def build_feature():
    # 将不同类型的节点统一到一个文本维度进行表征
    result = pd.read_csv('../Dataset/HGAT_train_data/HGAT_data.csv', lineterminator="\n")
    data = result[["tweet_id", "text\r"]]
    print(data.shape)
    # 按行遍历数据
    # 假设传入的也是类似于从数据库中读取的json格式
    ans = []
    for index, row in data.iterrows():
        info = {
            "tweet_id": int(row["tweet_id"]),
            "text": row["text\r"],
        }
        ans.append(info)
    # 获取实体和words的表征

