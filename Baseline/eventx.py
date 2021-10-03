#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
实现eventX方法
"""
import sys
import networkx as nx
import itertools
import random
from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.twitter_preprocessor import *
# 初始化分词实例
Cut = TwitterPreprocessor()


def construct_dict(doc):
    # contains all pairs(sorted) of keywords/entities and their corresponding enclosing tweets' ids as values
    kw_pair_dict = {}
    # contains all distinct keywords and entities as keys, and their corresponding enclosing tweets' ids as values
    kw_dict = {}
    m_tweets = []
    # 标签数据
    ground_truths = []
    # 按行遍历数据
    for d in doc:
        tweet_id = d['_id']
        words = Cut.get_token(d['text'])
        entities = Cut.entity_recognition(d['text'])
        entities = ['_'.join(tup) for tup in entities]
        for each in entities:
            if each not in kw_dict.keys():
                kw_dict[each] = []
            kw_dict[each].append(tweet_id)
        for each in words:
            if each not in kw_dict.keys():
                kw_dict[each] = []
            kw_dict[each].append(tweet_id)
        # for r in itertools.product(entities, words):
        for r in itertools.combinations(entities + words, 2):
            r = list(r)
            r.sort()
            pair = (r[0], r[1])
            if pair not in kw_pair_dict.keys():
                kw_pair_dict[pair] = []
            kw_pair_dict[pair].append(tweet_id)
        ground_truths.append(d['event_id'])
        m_tweets.append(entities + words)
    return kw_pair_dict, kw_dict, ground_truths, m_tweets


def map_dicts(kw_pair_dict, kw_dict):
    """
    构建keywords和enenties的索引，以便于后续生成TF-IDF嵌入
    :param kw_pair_dict:
    :param kw_dict:
    :param dir_path:
    :return:
    """
    map_index_to_kw = {}
    m_kw_dict = {}
    for i, k in enumerate(kw_dict.keys()):
        map_index_to_kw['k' + str(i)] = k
        m_kw_dict['k' + str(i)] = kw_dict[k]
    map_kw_to_index = {v: k for k, v in map_index_to_kw.items()}
    m_kw_pair_dict = {}
    for _, pair in enumerate(kw_pair_dict.keys()):
        m_kw_pair_dict[(map_kw_to_index[pair[0]], map_kw_to_index[pair[1]])] = kw_pair_dict[pair]
    return m_kw_pair_dict, m_kw_dict, map_index_to_kw, map_kw_to_index


def construct_kw_graph(kw_pair_dict, kw_dict, min_cooccur_time, min_prob):
    """
    construct keyword graph.
    Use all keywords as nodes and add edges between pairs that met the above two conditions
    :param kw_pair_dict:
    :param kw_dict:
    :param min_cooccur_time:  the times of co-occurrence shall be above a minimum threshold min_cooccur_time
    :param min_prob: the conditional probabilities of the occurrence Pr{wj|wi} and Pr{wi|wj} also need to be greater
                    than a predefined threshold min_prob
    :return: Graph
    """
    G = nx.Graph()
    # add nodes
    G.add_nodes_from(list(kw_dict.keys()))
    # add edges between pairs of keywords that can meet the 2 conditions
    for pair, co_tid_list in kw_pair_dict.items():
        if (len(co_tid_list) > min_cooccur_time):
            # print('condition 1 met')
            # print(pair, co_tid_list)
            if (len(co_tid_list) / len(kw_dict[pair[0]]) > min_prob) and (
                    len(co_tid_list) / len(kw_dict[pair[1]]) > min_prob):
                # print('condition 2 met')
                # print(pair[0], kw_dict[pair[0]])
                # print(pair[1], kw_dict[pair[1]])
                G.add_edge(*pair)
            # print()

    return G


def detect_kw_communities_iter(G, communities, kw_pair_dict, kw_dict, max_kw_num=3):
    """
    递归版本
    :param G:
    :param communities:
    :param kw_pair_dict:
    :param kw_dict:
    :param max_kw_num:   the splitting process ends if the number of nodes in each subgraph
                        is smaller than a predefined threshold max_kw_num
    :return:
    """
    connected_components = [c for c in nx.connected_components(G)]
    while len(connected_components) >= 1:
        c = connected_components[0]
        if len(c) < max_kw_num:
            communities.append(c)
            G.remove_nodes_from(c)
        else:
            c_sub_G = G.subgraph(c).copy()
            d = nx.edge_betweenness_centrality(c_sub_G)
            max_value = max(d.values())
            edges = [key for key, value in d.items() if value == max_value]
            # If two edges have the same betweenness score, the one with lower conditional probability will be removed
            if len(edges) > 1:
                probs = []
                for e in edges:
                    e = list(e)
                    e.sort()
                    pair = (e[0], e[1])
                    co_len = len(kw_pair_dict[pair])
                    e_prob = (co_len / len(kw_dict[pair[0]]) + co_len / len(kw_dict[pair[1]])) / 2
                    probs.append(e_prob)
                min_prob = min(probs)
                min_index = [i for i, j in enumerate(probs) if j == min_prob]
                edge_to_remove = edges[min_index[0]]
            else:
                edge_to_remove = edges[0]
            G.remove_edge(*edge_to_remove)
        connected_components = [c for c in nx.connected_components(G)]


def detect_kw_communities(G, communities, kw_pair_dict, kw_dict, max_kw_num=3):
    """
    迭代版本：递归版本在大图上运行时可能导致RecursionError
    :param G:
    :param communities:
    :param kw_pair_dict:
    :param kw_dict:
    :param max_kw_num:   the splitting process ends if the number of nodes in each subgraph
                        is smaller than a predefined threshold max_kw_num
    :return:
    """
    connected_components = [c for c in nx.connected_components(G)]
    if len(connected_components) >= 1:
        c = connected_components[0]
        if len(c) < max_kw_num:
            communities.append(c)
            G.remove_nodes_from(c)
        else:
            c_sub_G = G.subgraph(c).copy()
            d = nx.edge_betweenness_centrality(c_sub_G)
            max_value = max(d.values())
            edges = [key for key, value in d.items() if value == max_value]
            # If two edges have the same betweenness score, the one with lower conditional probability will be removed
            if len(edges) > 1:
                probs = []
                for e in edges:
                    e = list(e)
                    e.sort()
                    pair = (e[0], e[1])
                    co_len = len(kw_pair_dict[pair])
                    e_prob = (co_len / len(kw_dict[pair[0]]) + co_len / len(kw_dict[pair[1]])) / 2
                    probs.append(e_prob)
                min_prob = min(probs)
                min_index = [i for i, j in enumerate(probs) if j == min_prob]
                edge_to_remove = edges[min_index[0]]
            else:
                edge_to_remove = edges[0]
            G.remove_edge(*edge_to_remove)
        detect_kw_communities(G, communities, kw_pair_dict, kw_dict, max_kw_num)
    else:
        return


def map_communities(communities, map_kw_to_index):
    """
    将每个关键字簇中的节点映射为编码格式，以便于用tf-idf表示后者
    :param communities:
    :param map_kw_to_index:
    :return:
    """
    m_communities = []
    for cluster in communities:
        m_cluster = ' '.join(map_kw_to_index[kw] for kw in cluster)
        m_communities.append(m_cluster)
    return m_communities


def classify_docs(test_tweets, m_communities, map_kw_to_index, dir_path=None):
    m_test_tweets = []
    for doc in test_tweets:
        # print(doc)
        m_doc = ' '.join(map_kw_to_index[kw] for kw in doc)
        m_test_tweets.append(m_doc)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(m_communities + m_test_tweets)
    # print(m_test_tweets)
    # a1 = cosine_similarity(m_test_tweets[0], X[0])
    # print(a1)
    train_size = len(m_communities)
    test_size = len(m_test_tweets)
    classes = []
    for i in range(test_size):
        # print(i)
        cosine_similarities = linear_kernel(X[train_size + i], X[:train_size]).flatten()
        max_similarity = cosine_similarities[cosine_similarities.argsort()[-1]]
        related_clusters = [i for i, sim in enumerate(cosine_similarities) if sim == max_similarity]
        if len(related_clusters) == 1:
            classes.append(related_clusters[0])
        else:
            classes.append(random.choice(related_clusters))

    # if dir_path is not None:
    #     np.save(dir_path + '/classes.npy', classes)
    return classes


def event_method(doc):
    """
    调用方法，给出聚类结果
    :param doc:
    :return:
    """
    # 固定参数信息
    min_cooccur_time = 2
    min_prob = 0.15
    max_kw_num = 3
    # 需要修改这里
    kw_pair_dict, kw_dict, ground_truths, m_tweets = construct_dict(doc)
    m_kw_pair_dict, m_kw_dict, map_index_to_kw, map_kw_to_index = map_dicts(kw_pair_dict, kw_dict, dir_path="../Data/X")
    # 构造图
    G = construct_kw_graph(kw_pair_dict, kw_dict, min_cooccur_time, min_prob)
    # 从图中检测社区
    communities = []
    # split the keyword graph into clusters (stored in communities, a list of lists of nodes that belong to the same cluster)
    # detect_kw_communities(G, communities, kw_pair_dict, kw_dict, max_kw_num = max_kw_num)
    detect_kw_communities_iter(G, communities, kw_pair_dict, kw_dict, max_kw_num=max_kw_num)
    m_communities = map_communities(communities, map_kw_to_index)
    classes = classify_docs(m_tweets, m_communities, map_kw_to_index, "../Data/X")
    NMI = metrics.normalized_mutual_info_score(ground_truths, classes)
    NMI = round(NMI, 3)
    ars = metrics.adjusted_rand_score(ground_truths, classes)
    ars = round(ars, 3)

