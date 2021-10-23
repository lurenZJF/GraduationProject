#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# 将数据存储到mongodb

import pymongo
import numpy as np
import pandas as pd
import datetime
# 链接到数据库
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# 英文数据处理
# 创建一个数据库
mydbE = myclient["TwitterEvent2012"]
# 创建一个集合
mycolE = mydbE["tweets"]
# 读取数据后，转化成json格式，存储到mongodb
# 读取数据
df = np.load('TwitterEvent2012.npy', allow_pickle=True)
data = pd.DataFrame(data=df, columns=["event_id", "tweet_id", "text", "created_at", "entity", "words"])
print(data.shape)
# 按行遍历数据
for index, row in data.iterrows():
    info = {
        "_id": int(row["tweet_id"]),
        "event_id": int(row["event_id"]),
        "text": row["text"],
        "created_at": datetime.datetime.strftime(row["created_at"], '%Y-%m-%d')
    }
    x = mycolE.insert_one(info)

# 总计多少条数据
print(mycolE.find().count())
""" """
"""
中文数据集处理

# 创建一个数据库
mydbC = myclient["NewsEvent"]
# 创建一个集合
mycolC = mydbC["news"]
# 读取数据后，转化成json格式，存储到mongodb
data = pd.read_csv("event_story.csv")
print(data.shape)
# 按行遍历数据
for index, row in data.iterrows():
    dateArray = datetime.datetime.utcfromtimestamp(int(row["time"]/1000))
    otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
    info = {
        "story_id": row['story_id'],
        "event_id": row["event_id"],
        "category": row["category"],
        "time": otherStyleTime,
        "keywords": row["keywords"],
        "main_keywords": row["main_keywords"],
        "ner": row["ner"],
        "ner_keywords": row["ner_keywords"],
        "content": row["content"]
    }
    x = mycolC.insert_one(info)
# 总计多少条数据
print(mycolC.find().count())
"""