#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
func: 构建训练模型的异构图
训练模型时从CSV中读取数据，并将处理后的结果保存；
"""
import sys
import pandas as pd
import numpy as np
import networkx
import gensim
import pickle as pkl
import json
import collections
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.twitter_preprocessor import *
Cut = TwitterPreprocessor()
print("loading Gensim.word2vec. ")
model = gensim.models.Word2Vec.load('../Static/word2vec/word2vec_gensim_5')


def sample(datapath, DATASETS, resample=False, trainNumPerClass = 50):
    """
    训练HGAT模型时使用，用来划分训练集、测试集、验证集
    :param datapath:
    :param DATASETS:
    :param resample: True or False 是否需要划分数据集
    :param trainNumPerClass: 每个类别采样的数据大小
    :return:
    """
    if resample:
        X = []
        Y = []
        # 读取数据集
        data = pd.read_csv(datapath+'{}'.format(DATASETS), lineterminator="\n")
        for index, row in data.iterrows():
            X.append([row["tweet_id"]])
            Y.append(row["event_id"])
        # 文本类别
        cateset = list(set(Y))
        catemap = dict()
        for i in range(len(cateset)):
            catemap[cateset[i]] = i
        Y = [catemap[i] for i in Y]
        # 转化为array格式
        X = np.array(X)
        Y = np.array(Y)
        trainNum = trainNumPerClass * len(catemap)
        print(trainNum)
        ind_train, ind_test = train_test_split(X,
                                               train_size=trainNum, random_state=1, )
        # 将刚才划分的测试集，再次划分为训练集和验证集
        ind_vali, ind_test = train_test_split(ind_test,
                                              train_size=(len(X) - trainNum) / trainNum , random_state=1, )
        train = sum(ind_train.tolist(), [])
        vali = sum(ind_vali.tolist(), [])
        test = sum(ind_test.tolist(), [])
        print(len(train), len(vali), len(test))
        alltext = set(train + vali + test)
        print("train: {}\nvali: {}\ntest: {}\nAllTexts: {}".format(len(train), len(vali), len(test), len(alltext)))
        # 保存数据
        with open(datapath + 'train.list', 'w') as f:
            f.write('\n'.join(map(str, train)))
        with open(datapath + 'vali.list', 'w') as f:
            f.write('\n'.join(map(str, vali)))
        with open(datapath + 'test.list', 'w') as f:
            f.write('\n'.join(map(str, test)))
    else:
        # 读取已经切分好的数据
        train = []
        vali = []
        test = []
        with open(datapath + 'train.list', 'r') as f:
            for line in f:
                train.append(line.strip())
        with open(datapath + 'vali.list', 'r') as f:
            for line in f:
                vali.append(line.strip())
        with open(datapath + 'test.list', 'r') as f:
            for line in f:
                test.append(line.strip())
        alltext = set(train + vali + test)
    return train, vali, test, alltext


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


def get_sentence_entity(content, save_path="../Dataset/HGAT_train_data/HGAT2entity.txt"):
    """
    获取文本实体信息
    :param content: 数据集
    :param save_path: 当train_mode=True时，需要给出文件正确的存储路径
    :return:
    """
    ans = []
    for doc in content:
        ind = doc["_id"]
        entity = Cut.entity_recognition(doc["text"])
        ans.append([ind, entity])
    print("开始写入文件")
    # 将运行结果写入文件
    with open(save_path, "w+") as f:
        for ind, entity in ans:
            f.write("{}\t{}\n".format(ind, entity))


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
        graph.add_edges_from([(ind, e[0]) for e in entities])
        if len(entities) == 0:
            noEntity.add(ind)
            graph.add_node(ind)
    print("text-entity done.")
    return graph, entitySet, noEntity


def load_labels(g):
    """
    训练模式时，加载数据标签
    :param g:
    :return:
    """
    data = pd.read_csv("../Dataset/HGAT_train_data/HGAT_data.csv", lineterminator="\n")
    for index, row in data.iterrows():
        if row["tweet_id"] not in g.nodes():
            g.add_node(row["tweet_id"])
        g.nodes[row["tweet_id"]]["type"] = row["event_id"]
    return g


def build_graph(entity_list):
    """
    构建图，并加载实体信息
    :param entity_list:
    :return:
    """
    g = networkx.Graph()  # 创建图
    g, entity_set, no_entity = load_text_entity(graph=g, entity_list=entity_list)
    load_labels(g)
    sim_min = 0.5
    top_k = 10
    el = list(entity_set)
    entity_edge = []
    cnt_no = 0
    cnt_yes = 0
    cnt = 0
    for i in range(len(el)):
        sim_list = []
        top_k_left = top_k
        for j in range(len(el)):
            if i == j:
                continue
            cnt += 1
            try:
                sim = model.wv.similarity(el[i].lower().strip(')'), el[j].lower().strip(')'))
                cnt_yes += 1
                if sim >= sim_min:
                    entity_edge.append((el[i], el[j], {'sim': sim}))
                    top_k_left -= 1
                else:
                    sim_list.append((sim, el[j]))
            except Exception as e:
                cnt_no += 1
        sim_list = sorted(sim_list, key=(lambda x: x[0]), reverse=True)
        # 如果没满足top_K条关系的话，再进行数据补充
        for i in range(min(max(top_k_left, 0), len(sim_list))):
            entity_edge.append((el[i], sim_list[i][1], {'sim': sim_list[i][0]}))

    print(cnt_yes, cnt_no)
    g.add_edges_from(entity_edge)
    # save the network
    with open('../Dataset/HGAT_train_data/model_network_sampled.pkl', 'wb') as f:
        pkl.dump(g, f)


def build_entity_feature_with_description(graph, datapath="../"):
    """

    :param graph:
    :param datapath:
    :return:
    """
    nodes_set = set(graph.nodes())
    entity_index = []
    corpus = []
    cnt = 0
    # 读取wiki关于实体的解释部分
    for i in range(40):
        filename = str(i).zfill(4)
        with open("../Static/wikiAbstract/" + filename, 'r', encoding="utf-8") as f:
            for line in f:
                ent, desc = line.strip('\n').split('\t')
                entity = ent.replace(" ", "_")
                # 如果wiki描述的实体在node中
                if entity in nodes_set:
                    print("here", entity)
                    if entity not in entity_index:
                        entity_index.append(entity)
                        cnt += 1
                    else:
                        print('error')
                    content = Cut.get_token(desc)
                    content = ' '.join([word for word in content if word.isalpha()])
                    corpus.append(content)
    print(len(corpus), len(entity_index))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)  # 将文本数据转换为计数的稀疏矩阵
    print("Entity feature shape: ", X.shape)
    transformer = TfidfTransformer()  # 独热码形式转化为TF—IDF形式
    tfidf = transformer.fit_transform(X)
    print("Caculated! Saving...")
    # 存储数据
    with open(datapath + "vectorizer_model.pkl", 'wb') as f:
        pkl.dump(vectorizer, f)
    with open(datapath + "transformer_model.pkl", 'wb') as f:
        pkl.dump(transformer, f)
    with open(datapath + "features_entity_descBOW.pkl", 'wb') as f:
        pkl.dump(X, f)
    with open(datapath + "features_entity_descTFIDF.pkl", 'wb') as f:
        pkl.dump(tfidf, f)
    with open(datapath + "features_entity_index_desc.pkl", 'wb') as f:
        pkl.dump(entity_index, f)
    print("done!")


def build_text_feature(datapath, entity_list):
    """

    :param datapath:
    :param entity_list:
    :return:
    """
    # 这里先把未替换的ind-content对存在字典中
    pre_replace = dict()
    index2ind = {}
    cnt = 0
    corpus = []
    involved_entity = set()
    # 加载或者重新划分数据集
    train, vali, test, alltext = sample(datapath='../Dataset/HGAT_train_data/', DATASETS="HGAT_data.csv", resample=False)
    # 在整个数据集上做的TF-IDF表征
    data = pd.read_csv(datapath + 'HGAT_data.csv', lineterminator="\n")
    for index, row in data.iterrows():
        if str(row["tweet_id"]) not in alltext:
            continue
        pre_replace[row["tweet_id"]] = row["text\r"].lower()
        row["text\r"] = pre_replace[row["tweet_id"]]
        corpus.append(row["text\r"])
        index2ind[cnt] = row["tweet_id"]
        cnt += 1
    print('train: {}\tvali: {}\ttest: {}'.format(len(train), len(vali), len(test)))
    print('num of all corpus: {}'.format(len(train + vali + test)))
    print("loading entities...")
    for ind, entity in entity_list:
        if ind not in pre_replace:
            continue
        entity = json.loads(entity)
        for ent in entity:
            ent = ent.replace(" ", '')
            involved_entity.add(ent)
    print("text preprocessing...")
    # 清洗数据集
    corpus = preprocess_corpus_notDropEntity(corpus, involved_entity=involved_entity)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    print("text feature transformed.")
    alllength = sum([len(sentence.split(' ')) for sentence in corpus])
    avg_length = alllength / len(corpus)
    print('avg of tokens: {:.1f}'.format(avg_length))
    vocab = set()
    for s in corpus:
        vocab.update(s.split(' '))
    print('involved entities: {}'.format(len(involved_entity)))
    print('vocabulary size: {}'.format(len(vocab)))
    with open(datapath + 'TextBoW_model.pkl', 'wb') as f:
        pkl.dump(vectorizer, f)
    with open(datapath + "features_BOW.pkl", 'wb') as f:
        pkl.dump(X, f)
    with open(datapath + "features_TFIDF.pkl", 'wb') as f:
        pkl.dump(tfidf, f)
    with open(datapath + "features_index.pkl", 'wb') as f:
        pkl.dump(index2ind, f)
    print(X.shape)


def build_topic_feature_sklearn(datapath='../', TopicNum=50, train=False):
    """
    mine the latent topics T
    :param datapath:
    :param TopicNum: Topic数量
    :param train: 训练模型(True) 或者 加载模型(False)
    :return:
    """
    data = pd.read_csv("../Dataset/HGAT_train_data/HGAT_data.csv", lineterminator="\n")  # 读取原始数据
    idxlist = data["tweet_id"]
    idxlist = np.array(idxlist).tolist()
    with open(datapath + 'doc_index_LDA.pkl', 'wb') as f:
        pkl.dump(idxlist, f)

    with open(datapath + "features_BOW.pkl", 'rb') as f:
        X = pkl.load(f)
        # vocabulary_ 的对照关系，读上面那个bow的模型就可以了
    if train:
        # 根据过滤后的数据信息X，挖掘话题信息
        alpha, beta = 0.1, 0.1
        lda = LatentDirichletAllocation(n_components=TopicNum, max_iter=1200,
                                        learning_method='batch', n_jobs=-1,
                                        doc_topic_prior=alpha, topic_word_prior=beta,
                                        verbose=1,
                                        )
        lda_feature = lda.fit_transform(X)
        with open(datapath + 'lda_model.pkl', 'wb') as f:
            pkl.dump(lda, f)
        with open(datapath + 'topic_word_distribution.pkl', 'wb') as f:
            pkl.dump(lda.components_, f)
    else:
        with open(datapath + 'lda_model.pkl', 'rb') as f:
            lda = pkl.load(f)
        lda_feature = lda.transform(X)
    with open(datapath + 'doc_topic_distribution.pkl', 'wb') as f:
        pkl.dump(lda_feature, f)


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
    print("# text_nodes: {}     # entity_nodes: {}     # topic_nodes: {}".format(
        len(text_nodes), len(entity_nodes), len(topic_nodes)))
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


def remove(graph, datapath="../Dataset/HGAT_train_data/", TopK_for_Topics=2):
    # 读取基本模型
    with open(datapath + "features_BOW.pkl", 'rb') as f:
        features_BOW = pkl.load(f)
    with open(datapath + "features_TFIDF.pkl", 'rb') as f:
        features_TFIDF = pkl.load(f)
    with open(datapath + "features_index.pkl", 'rb') as f:
        features_index_BOWTFIDF = pkl.load(f)
    with open(datapath + "features_entity_descBOW.pkl", 'rb') as f:
        features_entity_BOW = pkl.load(f)
    with open(datapath + "features_entity_descTFIDF.pkl", 'rb') as f:
        features_entity_TFIDF = pkl.load(f)
    with open(datapath + "features_entity_index_desc.pkl", 'rb') as f:
        features_entity_index_desc = pkl.load(f)
    text_nodes, entity_nodes, topic_nodes = cnt_nodes(graph)
    feature = features_TFIDF
    features_index = features_index_BOWTFIDF
    entityF = features_entity_TFIDF
    features_entity_index = features_entity_index_desc
    textShape = feature.shape
    entityShape = entityF.shape
    print("Shape of text feature:", textShape, 'Shape of entity feature:', entityShape)
    # 删掉没有特征的实体
    notinind = set()
    entitySet = set(entity_nodes)
    print(len(entitySet))
    for i in entitySet:
        if i not in features_entity_index:
            notinind.add(i)
    print(len(graph.nodes()), len(notinind))
    graph.remove_nodes_from(notinind)
    entitySet = entitySet - notinind
    print(len(entitySet), len(features_entity_index))
    N = len(graph.nodes())
    print(len(graph.nodes()), len(graph.edges()))
    text_nodes, entity_nodes, topic_nodes = cnt_nodes(graph)
    # 删掉一些边
    cnt = 0
    nodes = graph.nodes()
    print(len(graph.edges()))
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
    print(len(graph.edges()), cnt)

    # 删掉孤立点（实体）
    delete = [n for n in graph.nodes() if len(graph[n]) == 0 and n not in text_nodes]
    print("Num of 孤立点：", len(delete))
    graph.remove_nodes_from(delete)

    train, vali, test, alltext = sample(datapath, DATASETS="HGAT_data.csv")
    # topic
    with open(datapath + 'topic_word_distribution.pkl', 'rb') as f:
        topic_word = pkl.load(f)
    with open(datapath + 'doc_topic_distribution.pkl', 'rb') as f:
        doc_topic = pkl.load(f)
    with open(datapath + 'doc_index_LDA.pkl', 'rb') as f:
        doc_idx_list = pkl.load(f)
    topic_num = topic_word.shape[0]
    topics = []
    for i in range(topic_num):
        topicName = 'topic_' + str(i)
        topics.append(topicName)
    topK_topics = naive_arg_topK(doc_topic, TopK_for_Topics, axis=1)
    for i in range(topK_topics.shape[0]):
        for j in range(TopK_for_Topics):
            graph.add_edge(doc_idx_list[i], topics[topK_topics[i, j]])
    print("gnodes:", len(graph.nodes()), "gedges:", len(graph.edges()))
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
    print('edges: ', cnt)
    # 存储数据
    text_nodes, entity_nodes, topic_nodes = cnt_nodes(graph)

    mapindex = dict()
    cnt = 0
    for i in text_nodes | entity_nodes | topic_nodes:
        mapindex[i] = cnt
        cnt += 1
    print(len(graph.nodes()), len(mapindex))
    outpath = '../Dataset/HGAT_train_data/out/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # 将训练集、验证集、测试集的索引存储起来
    train = list(map(int, train))
    test = list(map(int, test))
    vali = list(map(int, vali))
    gnodes = set(graph.nodes())
    print(gnodes, mapindex)
    with open(outpath + 'train.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in train if i in gnodes]))
    with open(outpath + 'vali.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in vali if i in gnodes]))
    with open(outpath + 'test.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in test if i in gnodes]))

    flag_zero = False

    input_type = 'text&entity&topic2hgcn'
    if input_type == 'text&entity&topic2hgcn':
        node_with_feature = set()
        DEBUG = False
        # text node
        content = dict()
        for i in range(textShape[0]):
            ind = features_index[i]
            if (ind) not in text_nodes:
                continue
            content[ind] = feature[i, :].toarray()[0].tolist()
            if DEBUG:
                content[ind] = feature[i, :10].toarray()[0].tolist()
            if flag_zero:
                entityFlen = entityShape[1]
                content[ind] += [0] * (entityFlen + topic_word.shape[1])
        data = pd.read_csv(datapath + 'HGAT_data.csv', lineterminator="\n")
        # 将事件类型加入

        for index, row in data.iterrows():
            if (row["tweet_id"]) not in text_nodes:
                continue
            content[row["tweet_id"]] += [row["event_id"]]
        # 存储数据
        with open(outpath + 'HGAT.content.text', 'w') as f:
            for ind in content:
                f.write(str(mapindex[ind]) + '\t' + '\t'.join(map(str, content[ind])) + '\n')
                node_with_feature.add(ind)
        cache = len(content)
        print("共{}个文本".format(len(content)))

        # entity node
        content = dict()
        for i in range(entityShape[0]):
            name = features_entity_index[i]
            if name not in entity_nodes:
                continue
            content[name] = entityF[i, :].toarray()[0].tolist() + ['entity']
            if flag_zero:
                content[name] = [0] * textShape[1] + content[name] + [0] * topic_word.shape[1] + ['entity']
        with open(outpath + 'HGAT.content.entity', 'w') as f:
            for ind in content:
                f.write(str(mapindex[ind]) + '\t' + '\t'.join(map(str, content[ind])) + '\n')
                node_with_feature.add(ind)
        cache += len(content)
        print("共{}个实体".format(len(content)))

        # topic node
        content = dict()
        for i in range(topic_num):
            #         zero_num = textShape[1] + entityFlen - topic_num
            topicName = topics[i]
            if topicName not in topic_nodes:
                continue
            one_hot = [0] * topic_num
            one_hot[i] = 1
            content[topicName] = one_hot
            content[topicName] = topic_word[i].tolist() + ['topic']
            if flag_zero:
                zero_num = textShape[1] + entityFlen
                content[topicName] = [0] * zero_num + content[topicName] + ['topic']

        # 存储TOPIC信息
        with open(outpath + 'HGAT.content.topic', 'w') as f:
            for ind in content:
                # print(len(content[ind]))
                f.write(str(mapindex[ind]) + '\t' + '\t'.join(map(str, content[ind])) + '\n')
                node_with_feature.add(ind)
        cache += len(content)
        print("共{}个主题".format(len(content)))

        print(cache, len(mapindex))
        print("nodes with features:", len(node_with_feature))

    # save mappings
    with open(outpath + 'mapindex.txt', 'w') as f:
        for i in mapindex:
            f.write("{}\t{}\n".format(i, mapindex[i]))

    # save adj matrix
    with open(outpath + 'HGAT.cites', 'w') as f:
        doneSet = set()
        nodeSet = set()
        for node in graphdict:
            for i in graphdict[node]:
                if (node, i) not in doneSet:
                    f.write(str(mapindex[node]) + '\t' + str(mapindex[i]) + '\n')
                    doneSet.add((i, node))
                    doneSet.add((node, i))
                    nodeSet.add(node)
                    nodeSet.add(i)
        for i in range(len(mapindex)):
            f.write(str(i) + '\t' + str(i) + '\n')
    print('Num of nodes with edges: ', len(nodeSet))


if __name__ == "__main__":
    # 划分数据集，方便后续训练使用
    # train, vali, test, alltext = sample(datapath='../Dataset/HGAT_train_data/',
    #                                     DATASETS="HGAT_data.csv",
    #                                     resample=True)
    # exit()
    # print('reading...')
    # result = pd.read_csv('../Dataset/HGAT_train_data/HGAT_data.csv', lineterminator="\n")
    # data = result[["tweet_id", "text\r"]]
    # print(data.shape)
    # 按行遍历数据
    # 假设传入的也是类似于从数据库中读取的json格式
    # ans = []
    # for index, row in data.iterrows():
    #     info = {
    #         "tweet_id": int(row["tweet_id"]),
    #         "text": row["text\r"],
    #     }
    #     ans.append(info)
    # 先将所有的数据 识别 实体
    # get_sentence_entity(ans, train_mode=True)
    # 加载数据识别的实体数据
    # ans = []
    # with open("../Dataset/HGAT_train_data/HGAT2entity.txt", "r") as f:
    #     for line in f:
    #         ind, entityList = line.strip('\n').split('\t')
    #         ans.append([ind, entityList])
    # build_graph(ans, train_model=True)
    # 加载实体关系图
    # with open('../Dataset/HGAT_train_data/model_network_sampled.pkl', "rb") as f:
    #     g = pkl.load(f)
    # build_entity_feature_with_description(graph=g, datapath="../Dataset/HGAT_train_data/", train_mode=True)
    # build_text_feature("../Dataset/HGAT_train_data/", ans, train_mode=True)
    # build_topic_feature_sklearn("../Dataset/HGAT_train_data/", train=True)
    # remove(graph=g)
    pass