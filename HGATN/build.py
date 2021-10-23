#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
func: 构建流式场景下的异构图
用HGAT表征文本时，无需保存处理后的数据，直接传入下一个函数；
"""
import sys
import pandas as pd
import networkx
import gensim
import collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
import pickle
import dgl
import datetime
from sklearn.decomposition import PCA
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.twitter_preprocessor import *
from HGAT.utils import *
from TextFiltering.stream import MONGO
# 加载分词模型
Cut = TwitterPreprocessor()
# 加载word2vec模型
model = gensim.models.Word2Vec.load('../Static/word2vec/word2vec_gensim_5')
# pca模型统一到200维
pca = PCA(n_components=200)


def normalizeF(mx):
    sup = np.absolute(mx).max()
    if sup == 0:
        return mx
    return mx / sup


def preprocess_corpus_notDropEntity(corpus, involved_entity):
    """
    过滤低频词、不在involved_entity中的词
    :param corpus:
    :param involved_entity:
    :return:
    """
    corpus2 = [Cut.get_token(sentence) for sentence in corpus]
    # 计算每个词的出现频率
    all_words = defaultdict(int)
    for c in corpus2:
        for w in c:
            all_words[w] += 1
    # 计算低频词
    low_freq = set(word for word in set(all_words) if all_words[word] < 5 and word not in involved_entity)
    # 去除低频词
    text = [[word for word in sentence if word not in low_freq] for sentence in corpus2]
    ans = [' '.join(i) for i in text]
    return ans


def get_sentence_entity(content):
    """
    获取文本实体信息
    :param content: 数据集
    :return:
    """
    ans = []
    for doc in content:
        ind = doc["_id"]
        entity = Cut.entity_recognition(doc["text"])
        entity = list(set(entity))
        ans.append([ind, entity])
    return ans


def load_text_entity(graph, entity_list):
    """
    图G中加载实体信息
    :param g:
    :return:
    """
    entitySet = set()
    noEntity = set()
    for ind, entity in entity_list:
        entities = [(d.replace(" ", '_')) for d in entity]
        entitySet.update([d.replace(" ", '_') for d in entity])
        # 添加一个edge数组
        graph.add_edges_from([(ind, e) for e in entities])
        if len(entities) == 0:
            noEntity.add(ind)
            graph.add_node(ind)
    return graph, entitySet, noEntity


def build_graph(entity_list):
    """
    构建图，并加载实体信息
    :param entity_list:
    :return:
    """
    g = networkx.Graph()  # 创建图
    g, entity_set, no_entity = load_text_entity(graph=g, entity_list=entity_list)
    sim_min = 0.5
    top_k = 10
    el = list(entity_set)
    entity_edge = []
    for i in range(len(el)):
        sim_list = []
        top_k_left = top_k
        for j in range(len(el)):
            if i == j:
                continue
            try:
                sim = model.wv.similarity(el[i], el[j])
                if sim >= sim_min:
                    entity_edge.append((el[i], el[j], {'sim': sim}))
                    top_k_left -= 1
                else:
                    sim_list.append((sim, el[j]))
            except Exception as e:
                pass
        sim_list = sorted(sim_list, key=(lambda x: x[0]), reverse=True)
        # 如果没满足top_K条关系的话，再进行数据补充
        for i in range(min(max(top_k_left, 0), len(sim_list))):
            entity_edge.append((el[i], sim_list[i][1], {'sim': sim_list[i][0]}))
    g.add_edges_from(entity_edge)
    return g


def build_entity_feature_with_description(graph):
    """

    :param graph:
    :return:
    """
    nodes_set = set(graph.nodes())
    entity_index = []
    corpus = []
    # 读取wiki关于实体的解释部分
    for i in range(40):
        filename = str(i).zfill(4)
        with open("../Static/wikiAbstract/" + filename, 'r', encoding="utf-8") as f:
            for line in f:
                ent, desc = line.strip('\n').split('\t')
                entity = ent.replace(" ", "_")
                entity = entity.lower()
                if entity in nodes_set:  # 如果wiki描述的实体在node中
                    if entity not in entity_index:
                        entity_index.append(entity)
                        content = Cut.get_token(desc)
                        content = ' '.join([word for word in content if word.isalpha()])
                        corpus.append(content)
                    else:
                        # 出现歧义
                        # print('error')
                        pass


    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)  # 将文本数据转换为计数的稀疏矩阵
    transformer = TfidfTransformer()  # 独热码形式转化为TF—IDF形式
    tfidf = transformer.fit_transform(X)
    return X, tfidf, entity_index


def build_text_feature(entity_list, doc=None):
    """

    :param entity_list:
    :param doc:
    :return:
    """
    # 这里先把未替换的ind-content对存在字典中
    pre_replace = dict()
    index2ind = {}
    cnt = 0
    corpus = []
    involved_entity = set()
    for d in doc:
        pre_replace[d["_id"]] = d["text"].lower()
        d["text"] = pre_replace[d["_id"]]
        corpus.append(d["text"])
        index2ind[cnt] = d["_id"]
        cnt += 1
    # print("loading entities...")
    for ind, entity in entity_list:
        if ind not in pre_replace:
            continue
        for ent in entity:
            ent = ent.replace(" ", '')
            involved_entity.add(ent)
    # 清洗数据集
    corpus = preprocess_corpus_notDropEntity(corpus, involved_entity=involved_entity)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    vocab = set()
    for s in corpus:
        vocab.update(s.split(' '))
    return X, tfidf, index2ind


def build_topic_feature_sklearn(TopicNum=50, X=None):
    """
    mine the latent topics T
    :param TopicNum: Topic数量
    :param X:
    :return:
    """
    alpha, beta = 0.1, 0.1
    lda = LatentDirichletAllocation(n_components=TopicNum, max_iter=200,
                                    learning_method='batch', n_jobs=-1,
                                    doc_topic_prior=alpha, topic_word_prior=beta
                                    )
    lda_feature = lda.fit_transform(X)
    return lda.components_, lda_feature


def cnt_nodes(g):
    """
    查看异构图中各节点数量
    :param g: 图
    :return: 返回图中各种节点数量
    """
    text_nodes, entity_nodes, topic_nodes = set(), set(), set()
    for i in g.nodes():
        if isinstance(i, int):
            text_nodes.add(i)
        elif i[:6] == "topic_":
            topic_nodes.add(i)
        else:
            entity_nodes.add(i)
    return text_nodes, entity_nodes, topic_nodes


def naive_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(-matrix, axis=axis)
    return full_sort.take(np.arange(K), axis=axis)


def remove(graph, features_TFIDF, features_index_BOWTFIDF, features_entity_TFIDF,
           features_entity_index_desc, topic_word, doc_topic, doc_idx_list,
           TopK_for_Topics=2):
    text_nodes, entity_nodes, topic_nodes = cnt_nodes(graph)
    feature = features_TFIDF
    features_index = features_index_BOWTFIDF
    entityF = features_entity_TFIDF
    features_entity_index = features_entity_index_desc
    textShape = feature.shape
    entityShape = entityF.shape
    notinind = set()  # 删掉没有特征的实体
    entitySet = set(entity_nodes)
    for i in entitySet:
        if i not in features_entity_index:
            notinind.add(i)
    # 删除部分节点
    graph.remove_nodes_from(notinind)
    # 重新计数
    text_nodes, entity_nodes, topic_nodes = cnt_nodes(graph)
    # 删掉相似度较小的边
    cnt = 0
    nodes = graph.nodes()
    for node in nodes:
        try:
            cache = [j for j in graph[node]
                     if ('sim' in graph[node][j] and graph[node][j]['sim'] < 0.5)  # 0.5
                     ]
            if len(cache) != 0:
                graph.remove_edges_from([(node, i) for i in cache])
            cnt += len(cache)
        except:
            print(graph[node])
            break
    # 删掉孤立点（实体）
    delete = [n for n in graph.nodes() if len(graph[n]) == 0 and n not in text_nodes]
    graph.remove_nodes_from(delete)
    # topic
    topic_num = topic_word.shape[0]
    topics = []
    for i in range(topic_num):
        topicName = 'topic_' + str(i)
        topics.append(topicName)
    topK_topics = naive_arg_topK(doc_topic, TopK_for_Topics, axis=1)
    for i in range(topK_topics.shape[0]):
        for j in range(TopK_for_Topics):
            graph.add_edge(doc_idx_list[i], topics[topK_topics[i, j]])
    # build Edges data
    cnt = 0
    nodes = graph.nodes()
    graphdict = collections.defaultdict(list)
    for node in nodes:
        try:
            cache = [j for j in graph[node]
                     if ('sim' in graph[node][j] and graph[node][j]['sim'] >= 0.5) or ('sim' not in graph[node][j])
                     # 0.5
                     ]
            if len(cache) != 0:
                graphdict[node] = cache
            cnt += len(cache)
        except:
            print(graph[node])
            break
    # print('edges: ', cnt)
    # 重新计数
    text_nodes, entity_nodes, topic_nodes = cnt_nodes(graph)
    mapindex = dict()
    cnt = 0
    for i in text_nodes | entity_nodes | topic_nodes:
        mapindex[i] = cnt
        cnt += 1
    # print(len(graph.nodes()), len(mapindex))
    # HIN包含三张类型的节点：文本、话题、实体
    type_list = ['text', 'topic', 'entity']
    # 构造map， key为节点类型，内容为一个集合
    idx2type = {t: set() for t in type_list}
    features_list = []
    idx_map_list = []
    for type_name in type_list:
        indexes, features = [], []
        if type_name == "text":
            for i in range(textShape[0]):
                ind = features_index[i]
                if (ind) not in text_nodes:
                    continue
                indexes.append(mapindex[ind])
                featureT = feature[i, :].toarray()[0]
                # 将feature特征统一到200维度
                featureT = pca.fit_transform(featureT)
                features.append(featureT)
        elif type_name == "entity":
            for i in range(entityShape[0]):
                name = features_entity_index[i]
                if name not in entity_nodes:
                    continue
                indexes.append(mapindex[name])
                featureE = entityF[i, :].toarray()[0]
                featureE = pca.fit_transform(featureE)
                features.append(featureE)
        else:
            for i in range(topic_num):
                topicName = topics[i]
                if topicName not in topic_nodes:
                    continue
                indexes.append(mapindex[topicName])
                features.append(topic_word[i])
        features = np.stack(features)
        features = normalize(features)
        features = torch.FloatTensor(features)
        features = dense_tensor_to_sparse(features)
        features_list.append(features)
        # 索引匹配
        idx = np.stack(indexes)
        for i in idx:
            idx2type[type_name].add(i)
        idx_map = {j: i for i, j in enumerate(idx)}
        idx_map_list.append(idx_map)
    # adj matrix
    doneSet = set()
    edges_unordered = []
    for node in graphdict:
        for i in graphdict[node]:
            if (node, i) not in doneSet:
                edges_unordered.append([mapindex[node], mapindex[i]])
    for i in range(len(mapindex)):
        edges_unordered.append([i, i])

    len_list = [len(idx2type[t]) for t in type_list]
    type2len = {t: len(idx2type[t]) for t in type_list}
    len_all = sum(len_list)
    adj_all = sp.lil_matrix(np.zeros((len_all, len_all)), dtype=np.float32)
    # 应该返回一个图
    adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]
    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):
            t1, t2 = type_list[i1], type_list[i2]
            if i1 == i2:
                edges = []
                for edge in edges_unordered:
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])])
                edges = np.array(edges)
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        shape=(type2len[t1], type2len[t2]), dtype=np.float32)
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil()

            elif i1 < i2:
                edges = []
                for edge in edges_unordered:
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])])
                    elif (edge[1] in idx2type[t1] and edge[0] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[1]), idx_map_list[i2].get(edge[0])])
                edges = np.array(edges)
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        shape=(type2len[t1], type2len[t2]), dtype=np.float32)
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)

                adj_all[
                sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil()
                adj_all[
                sum(len_list[:i2]): sum(len_list[:i2 + 1]),
                sum(len_list[:i1]): sum(len_list[:i1 + 1])] = adj.T.tolil()

    adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
    adj_all = normalize_adj(adj_all + sp.eye(adj_all.shape[0]))
    adj1 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:1]): sum(len_list[:2])]
    adj2 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:2]): sum(len_list[:3])]
    adj3 = adj_all[sum(len_list[:1]): sum(len_list[:2]), sum(len_list[:1]): sum(len_list[:2])]

    return features_list, adj_list


def build_information_network(res):
    """
    这里的res应该是什么样子的呢？是一个json还是一个默认的游标？
    :param res:
    :return:
    """
    # 先将所有的数据 识别 实体
    # ans 中的第一列为id信息，第二列为实体信息
    print("step1")
    ans = get_sentence_entity(res)
    # 加载实体信息，构建图
    print("step2")
    graph = build_graph(ans)
    print("step3")
    features_entity_descBOW, features_entity_descTFIDF, features_entity_index_desc = \
        build_entity_feature_with_description(graph=graph)
    # text_feature
    print("step4")
    features_BOW, features_TFIDF, features_index = build_text_feature(ans, res)
    print("step5")
    lda_model_word, doc_topic_distribution = build_topic_feature_sklearn(X=features_BOW)
    # 生成异构图
    ans = np.array(ans)
    print("step6")
    FL, AL = remove(graph, features_TFIDF,
                    features_index_BOWTFIDF=features_index,
                    features_entity_TFIDF=features_entity_descTFIDF,
                    features_entity_index_desc=features_entity_index_desc,
                    topic_word=lda_model_word,
                    doc_topic=doc_topic_distribution,
                    doc_idx_list=ans[:, 0],
                    TopK_for_Topics=2)
    return FL, AL


def stream_information(res):
    ans = []
    # cnt = 1
    for obj in res:
        info = {
                    "_id": int(obj["_id"]),
                    "text": obj["text"]
                }
        # cnt += 1
        # if cnt >= 2000:
        #     break
        ans.append(info)
    # print("数据样本量：", cnt)
    FL, AL = build_information_network(ans)
    for a in AL:
        print(a.shape)
    # print(FL.shape, AL.shape)
    # for i in range(3):
    #     print(FL[i].shape)
    # path = "../Dataset/HGAT_train_data/out/"
    # AL, FL, labels, idx_train_ori, idx_val_ori, idx_test_ori, idx_map = load_data(path=path)
    # FL = np.array(FL)
    # AL = np.array(AL)
    # np.savez("raw_FL.npz")
    # np.savez("raw_AL.npz")
    # # print(FL.shape, AL.shape)
    # output = HMODEL(FL, AL)
    # embedding = HMODEL.emb2.tolist()
    # 计算距离
    # X = pairwise_distances(embedding)
    # exit(print("here"))
    # return AL


def test_data():
    result = pd.read_csv('../Dataset/HGAT_train_data/HGAT_data.csv', lineterminator="\n")
    data = result[["tweet_id", "text\r"]]
    ans = []
    for index, row in data.iterrows():
        info = {
            "_id": int(row["tweet_id"]),
            "text": row["text\r"],
        }
        ans.append(info)
    print(len(ans))
    exit()
    return stream_information(ans)


# if __name__ == "__main__":
#     result = pd.read_csv('../Dataset/HGAT_train_data/HGAT_data.csv', lineterminator="\n")
#     data = result[["tweet_id", "text\r"]]
#     data = data[:200]
#     print(data.shape)
#     # 按行遍历数据
#     # 假设传入的也是类似于从数据库中读取的json格式
#     ans = []
#     for index, row in data.iterrows():
#         info = {
#             "_id": int(row["tweet_id"]),
#             "text": row["text\r"],
#         }
#         ans.append(info)
#     FL1, AL1 = build_information_network(ans)
#     FL2 = np.array(FL1)
#     AL2 = np.array(AL1)
#     print(FL2.shape)
#     print(AL2.shape)
#     for f in FL2:
#         print(f.shape)
#     for f in AL2:
#         print(f.shape)
#     # 保存两个np
#     # np.savez("adj_list.npz", AL2)
#     # np.savez("feature_list.npz", FL2)


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
    time_list = range_date("2012-10-10", "2012-11-07")
    N = len(time_list)
    # 调用数据查询方法
    MG = MONGO("TwitterEvent2012", "tweets")
    for i in range(N - 1):
        # 根据时间信息进行信息检索；返回的是游标，会随着遍历而移动
        res = MG.query(time_list[i], time_list[i + 1])
        stream_information(res)







