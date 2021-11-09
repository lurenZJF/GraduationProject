#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
实现事件时序合并算法
"""
import numpy as np
from collections import Counter
from summa import keywords, summarizer
from nltk.corpus import stopwords
import sys
import datetime
import time
import os
path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(path+"/")
from TextFiltering.stream import ES
es = ES(database="TwitterEvent2012", collection="event7")


def word2tf(word_list1, word_list2):
    """
    词袋模型
    :param word_list1:
    :param word_list2:
    :return:
    """
    # 转成counter不需要考虑0的情况
    words1_dict = Counter(word_list1)
    words2_dict = Counter(word_list2)
    bags = set(words1_dict.keys()).union(set(words2_dict.keys()))
    # 转成list对debug比较方便吗，防止循环集合每次结果不一致
    bags = sorted(list(bags))
    vec_words1 = [words1_dict[i] for i in bags]
    vec_words2 = [words2_dict[i] for i in bags]
    # 转numpy
    vec_words1 = np.asarray(vec_words1, dtype=np.float)
    vec_words2 = np.asarray(vec_words2, dtype=np.float)
    return vec_words1, vec_words2


def cosine_similarity(v1, v2):
    """
    计算余弦相似度
    :param v1:
    :param v2:
    :return:
    """
    # 余弦相似度
    v1, v2 = np.asarray(v1, dtype=np.float), np.asarray(v2, dtype=np.float)
    up = np.dot(v1, v2)
    down = np.linalg.norm(v1) * np.linalg.norm(v2)
    return round(up / down, 3)


def event_extract(event):
    """
    根据传入的事件，提取关键词和关键句
    :param event: 聚类得到的事件
    {
        "counts": 事件中文本的数量
        "source": 原始数据,
        "core_points": 核心点文本[sentence1,sentence2,...],
        "event_time": 事件时间
    }
    :return: event
    {
        "counts":
        "source":
        "core_points":
        "keywords": list,[word1,word2,...]
        "summary": list,[sentence1,sentence2]
        "event_time":
    }
    """
    text_num = int(event['counts'])
    if text_num > 300:  # 如果某类的事物太多
        # 选取其中的核心句作为关键词提取
        data = event["core_points"]
    else:
        data = event["source"]
    k_num = len(data)//50 + 3
    # 拼凑新的文本,提取关键词
    sentence = ""
    for obj in data:
        content = obj["text"]
        if content[-1] not in ['?', '!', ';', '？', '！', '。', '；', '…']:
            content = content + '。'  # 用句号将不同的句子进行隔
        sentence += content
    key = keywords.keywords(sentence, ratio=0.8, split=True, additional_stopwords=stopwords.words('english'))
    if len(key) > k_num:
        key = key[:k_num]
    # 获取主题中心句
    if len(event["core_points"]) < 5:  # 如果核心句数量太少，选择原始文本提取核心句；
        sentence_data = event["source"]
    else:
        sentence_data = event["core_points"]
    text = ""
    for obj in sentence_data:
        text += obj["text"]
    sums = summarizer.summarize(text, additional_stopwords=stopwords.words('english'))
    event["keywords"] = key
    event["summary"] = sums
    return event


def event_merge(event):
    """
    在给定时间范围内的事件簇中，进行事件合并
    :param event: 聚类得到的事件
    {
        "counts": 事件中文本的数量
        "source": 原始数据,
        "core_points": 核心点文本[sentence1,sentence2,...],
        "event_time": 事件时间
    }
    :return:
    """
    # 时间传话
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    key_words = event['keywords']
    today = strptime(event["event_time"], "%Y-%m-%d")
    tomorrow = today + datetime.timedelta(days=1)
    yesterday = today - datetime.timedelta(days=2)
    # 当天的时间进行合并
    # print(event["event_time"])
    # print(strftime(tomorrow))
    today_result = es.get_events_filter_by_keywords_and_date(keywords_list=key_words, start_time=event["event_time"],
                                                             end_time=strftime(tomorrow, "%Y-%m-%d"))
    # 昨天和前天的事件可以进行链接
    if today_result.count() > 0:  # 有符合条件的数据
        # 生成一个存储相似度的列表
        event_list = []
        index_list = []
        for e in today_result:
            index_list.append(e['_id'])
            event_list.append(e)
        sim_list = np.zeros(len(event_list))
        # 符合关键词条件的数据，计算相似度
        for i in range(len(event_list)):
            sim_list[i] = cosine_similarity(*word2tf(key_words, event_list[i]['keywords']))
            # 选取相似度最大的值
        sim_max = np.max(sim_list)
        if sim_max >= 0.7:
            index = np.where(sim_list == sim_max)[0][0]  # 索引信息
            # 将新的类和原有的合并
            clustered_sentences = event['source']
            clustered_sentences.extend(event_list[index]['source'])
            clustered_core = event['core_points']
            clustered_core.extend(event_list[index]['core_points'])
            clustered_keywords = event['keywords']
            clustered_keywords.extend(event_list[index]['keywords'])
            clustered_keywords = list(set(clustered_keywords))
            summary = event['summary']  # 摘要句不发生变化
            event_time = []
            event_time.append(event['event_time'])
            event_time.append(event_list[index]['event_time'])
            # 更新event
            event = {
                "counts": len(clustered_sentences),
                "source": clustered_sentences,
                "core_points": clustered_core,
                "keywords": clustered_keywords,
                "summary": summary,
                "event_time": min(event_time),
                "create_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                'is_merged': 0
            }
            # 将原有的事件更新为不可被查询
            es.update_doc(id=index_list[index])
    # 如果有合并，这里的event也有更新了
    yesterday_result = es.get_events_filter_by_keywords_and_date(keywords_list=key_words,
                                                                 start_time=yesterday, end_time=event["event_time"])
    if yesterday_result.count() > 0:
        # 生成一个存储相似度的列表
        event_list = []
        index_list = []
        for e in yesterday_result:
            index_list.append(e['_id'])
            event_list.append(e['_source'])
        sim_list = np.zeros(len(event_list))
        # 符合关键词条件的数据，计算相似度
        for i in range(len(event_list)):
            sim_list[i] = cosine_similarity(*word2tf(key_words, event_list[i]['keywords']))
        # 选取相似度最大的值
        sim_max = np.max(sim_list)
        if sim_max >= 0.6:
            index = np.where(sim_list == sim_max)[0][0]  # 索引信息
            # 新增old_keywords和old_id
            event["old_id"] = index_list[index]
    # 将更新后的事件存入数据库
    es.insert_event(event)







