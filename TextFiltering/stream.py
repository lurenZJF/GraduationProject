#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
func: 模拟数据流式传输
"""
import pymongo


class MONGO(object):
    def __init__(self, database, collection):
        # 连接数据库
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        # 选择需要使用的数据库
        self.database = self.client[database]
        # 选择数据库中的集合
        self.collection = self.database[collection]

    def query(self, start_time, end_time):
        """
        从集合中查询数据（start<=x<end）
        :param start_time: 起始时间
        :param end_time: 结束时间
        :return: json数据
        """
        res = self.collection.find(
            {
                "created_at":{"$gte":start_time,
                              "$lt":end_time}
            }
        )
        return res


# MG = MONGO("TwitterEvent2012", "tweets")
# res = MG.query('2012-10-22', '2012-10-23') # type: #Cursor
# for r in res:
#     print(r)
#     exit()