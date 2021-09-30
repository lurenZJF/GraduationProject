#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import datetime
import logging
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.stream import MONGO
# 日志信息
log_console = logging.StreamHandler(sys.stdout)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.INFO)
default_logger.addHandler(log_console)



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


if __name__ == "__main__":
    # 生成时间信息
    time_list= range_date("2012-10-10", "2012-11-07")
    N = len(time_list)
    # 调用数据查询方法
    MG = MONGO("TwitterEvent2012", "tweets")
    for i in range(N-1):
        # 根据时间信息进行信息检索
        res = MG.query(time_list[i], time_list[i+1])
        for r in res:
            print(r)
        exit()