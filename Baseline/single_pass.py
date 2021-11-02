#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实现single Pass聚类，验证模型效果
"""
import datetime
import time
import sys
import numpy as np
from math import sqrt
from sklearn import metrics
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.stream import MONGO
from Detect.utils import build_data
# from Baseline.glove2vec import *
# from TextFiltering.twitter_preprocessor import *
# 初始化分词实例
# Cut = TwitterPreprocessor()
# G = GenerateWordVectors("../Static/glove2word2vec.txt")


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


def manyVectorDistance(vec_a, vec_b, distance_type="Euclidean"):
    """
    根据距离类型不同 求两个向量的距离(暂有欧式距离/余弦距离)
    :param vec_a:
    :param vec_b:
    :param distance_type: 默认是欧式距离
    :return:
    """
    try:
        # 欧式距离
        if distance_type == "Euclidean":
            # 计算向量a与向量b的欧式距离
            diff = vec_a - vec_b
            # dot计算矩阵内积
            return sqrt(np.dot(diff, diff))
        # 余弦距离
        elif distance_type == "Cosine":
            # np.linalg.norm 矩阵范数  默认二范数  各项平方的和 再开根号
            return np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b))
    except TypeError:
        print("vec_a=%s" % vec_a)
        print("vec_b=%s" % vec_b)
        return None


# 定义一个簇单元
class ClusterUnit:
    def __init__(self):
        self.node_list = []  # 该簇包含的结点列表
        self.title_list = []  # 该簇中包含的结点的文本内容
        self.node_num = 0  # 簇中结点个数
        self.centroid = None  # 簇质心

    def addNode(self, node=0, node_vec=None, title=None):
        """
        为本簇添加指定结点，并更新簇质心
        :param node: 结点
        :param node_vec: 结点特征向量
        :return:
        """
        self.node_list.append(node)
        self.title_list.append(title)
        try:
            # 更新质心
            self.centroid = (self.node_num * self.centroid + node_vec) / (self.node_num + 1)
        except TypeError:
            # 初始化质心
            self.centroid = np.array(node_vec) * 1
        self.node_num += 1

    def removeNode(self, node=0, node_vec=None, title=None):
        try:
            self.node_list.remove(node)
            self.title_list.remove(title)
            try:
                self.centroid = (self.node_num * self.centroid - node_vec) / (self.node_num - 1)
            except ZeroDivisionError:
                self.centroid = None
            self.node_num -= 1
        except ValueError:
            # 该结点不在簇中
            raise ValueError("%s not in this cluster" % node)

    def moveNode(self, node, node_vec, title, another_cluster):
        # 移除本簇一个结点，到另一个簇中
        self.removeNode(node=node, node_vec=node_vec, title=title)
        another_cluster.addNode(node=node, node_vec=node_vec, title=title)

    def printNode(self):
        print("簇中结点个数为:%s，簇质心为:%s" % self.node_num, self.centroid)
        print("各个结点如下:\n")
        for title in self.title_list:
            print(title)


class SinglePassCluster:
    def __init__(self, threshold_list=None, vector_list=None, ids_list=None, title_list=None):
        """
        :param t:一趟聚类的阈值
        :param vector_list:
        """
        self.threshold_list = threshold_list  # 一趟聚类的阈值
        self.threshold = 2
        self.vector_list = np.array(vector_list)  # 存储所有文章的特征向量
        self.id_list = ids_list  # 存储所有文章的id, 与特征向量相对应
        self.title_list = title_list  # 存储所有文章的内容
        self.cluster_list = []  # 聚类后簇的列表
        t1 = time.time()
        self.clustering()
        t2 = time.time()
        self.cluster_num = len(self.cluster_list)  # 聚类完成后  簇的个数
        self.spend_time = t2-t1  # 一趟聚类花费的时间

    def clustering(self):
        # 初始新建一个簇
        self.cluster_list.append(ClusterUnit())
        # 读入的第一个文章（结点）归入第一个簇中
        self.cluster_list[0].addNode(node=self.id_list[0], node_vec=self.vector_list[0], title=self.title_list[0])
        length = len(self.id_list)
        # 遍历所有的文章  开始进行聚类  index 从1->(len-1)
        for index in range(length)[1:]:
            if self.threshold_list is not None:
                if self.threshold < (self.threshold_list[1]+self.threshold_list[0]) * 0.5:
                    self.threshold = index*(self.threshold_list[1] - self.threshold_list[0])/100
                    print("threshold is %s" % self.threshold)
                else:
                    self.threshold = (self.threshold_list[1] + self.threshold_list[0]) * 0.3
                    print("the last threshold is %s" % self.threshold)
            current_vector = self.vector_list[index]
            if current_vector is None:
                print("index=%s" % index)
                print("len(vectors)=%s" % len(self.vector_list))
            # 与簇的质心的最小距离
            min_distance = manyVectorDistance(distance_type="Euclidean", vec_a=current_vector,
                                              vec_b=self.cluster_list[0].centroid)
            # 最小距离的簇的索引
            min_cluster_index = 0
            for cluster_index, one_cluster in enumerate(self.cluster_list[1:]):
                # enumerate会将数组或列表组成一个索引序列
                # 寻找距离最小的簇，记录下距离和对应的簇的索引
                distance = manyVectorDistance(distance_type="Euclidean", vec_a=current_vector,
                                              vec_b=one_cluster.centroid)
                try:
                    if distance < min_distance:
                        min_distance = distance
                        # 因为cluster_index是从0开始
                        min_cluster_index = cluster_index + 1
                except TypeError:
                    print(distance)
            # 最小距离小于阈值，则归于该簇
            if min_distance < self.threshold:
                self.cluster_list[min_cluster_index].addNode(node=self.id_list[index], node_vec=current_vector,
                                                             title=self.title_list[index])
            else:
                new_cluster = ClusterUnit()
                new_cluster.addNode(node=self.id_list[index], node_vec=current_vector, title=self.title_list[index])
                self.cluster_list.append(new_cluster)
                del new_cluster

    def printClusterResult(self, label_dict=None):
        # 打印出聚类结果
        # label_dict:节点对应的标签字典
        print("**********single-pass cluster result******")
        for index, one_cluster in enumerate(self.cluster_list):
            # print("cluster_index:%s" % index)
            # 簇的结点列表
            # print(one_cluster.node_list)
            if label_dict is not None:
                # 若有提供标签字典，则输出该簇的标签
                print(" ".join([label_dict[n] for n in one_cluster.node_list]))
                print("node num:%s" % one_cluster.node_num)
                print("========================")
        print("the number of nodes %s" % len(self.vector_list))
        print("the number of cluster %s" % self.cluster_num)
        print("spend time %.9fs" % (self.spend_time / 1000))

    def saveClusterResult(self):
        """
        把各个聚类结果写到各个文件中
        :return:
        """
        print("the number of cluster %s" % self.cluster_num)
        print("spend time of cluster %.9fs" % (self.spend_time / 1000))
        for index, one_cluster in enumerate(self.cluster_list):
            if one_cluster.node_num < 7:
                continue
            if one_cluster.node_num < 3:
                cluster_write = open("result/22cluster%s.txt" % index, mode='w+', encoding='utf-8')
            elif one_cluster.node_num < 4:
                cluster_write = open("result/33cluster%s.txt" % index, mode='w+', encoding='utf-8')
            else:
                cluster_write = open("result/cluster%s.txt" % index, mode='w+', encoding='utf-8')
            cluster_write.write("类别%s情况如下：\n" % index)
            cluster_write.write("共有%s篇文章\n" % one_cluster.node_num)
            cluster_write.write("类别质心向量如下:\n%s" % one_cluster.centroid)
            cluster_write.write("\n文章id和内容如下:\n")
            for i, id in enumerate(one_cluster.node_list):
                cluster_write.write("%s %s\n" % (id, one_cluster.title_list[i]))

            cluster_write.close()

    def metric(self):
        print("the number of cluster %s" % self.cluster_num)
        print("spend time of cluster: %f" % self.spend_time)
        predict = []
        labels = []
        for index, one_cluster in enumerate(self.cluster_list):
            for i, id in enumerate(one_cluster.node_list):
                predict.append(index)
                labels.append(id)
        # print(predict)
        # print(labels)
        # print(len(predict))
        # print(len(labels))
        # 计算各项指标
        NMI = metrics.normalized_mutual_info_score(labels, predict)
        NMI = round(NMI, 3)
        ars = metrics.adjusted_rand_score(labels, predict)
        ars = round(ars, 3)
        # return [NMI, ars]
        print(NMI, ars)



if __name__ == "__main__":
    print("loading data")
    # t1 = time.time()
    time_list = range_date("2012-10-10", "2012-11-07")
    MG = MONGO("TwitterEvent2012", "tweets")
    res = MG.query(time_list[0], time_list[-1])
    contents, time_info, labels_true = build_data(res)
    # 文本表征
    print("learn embeddings")
    # token_w = []
    # for c in contents:
    #     words = Cut.get_token(c)
    #     token_w.append(words)
    # embedding = G.distance_matrix(token_w, dis=False)
    # np.savez('single_embedding.npz', embedding=embedding)
    data = np.load('single_embedding.npz')
    embedding = data["embedding"]
    # 对contents应用single-Pass
    print("start cluster")
    ans = SinglePassCluster(ids_list=labels_true, vector_list=embedding, title_list=contents)
    ans.metric()
    # print("聚类耗时：", ans.spend_time)
    # print("耗时：", time.time()-t1)