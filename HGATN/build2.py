#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
func: 构建流式场景下的异构图
将entity和words统一用word2vec表征，将Topic实验不同数据量的文本能否TOPIC维度保持一致
"""
import sys
import networkx
import gensim
import collections
import dgl
from sklearn.model_selection import train_test_split
import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
sys.path.append('/home/dell/GraduationProject/')
from TextFiltering.twitter_preprocessor import *
from HGAT.utils import *
from Baseline.glove2vec import *
# from HAN.model import *
from TextFiltering.stream import MONGO
# 加载分词模型
Cut = TwitterPreprocessor()
G = GenerateWordVectors("../Static/glove2word2vec.txt")
# 加载word2vec模型
print("???")
model = gensim.models.Word2Vec.load('../Static/word2vec/word2vec_gensim_5')
# 加载训练好的模型
# HMODEL = HAN(num_meta_paths=2, in_size=200, hidden_size=8, out_size=364, num_heads=[8], dropout=0.5)
# HMODEL.load_state_dict(torch.load("../HAN/early_stop_2021-10-20_15-07-15.pth"))
print("load end")


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
    return text, ans


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
                        corpus.append(content)
                    else:
                        # 出现歧义
                        # print('error')
                        pass
    embedding = G.distance_matrix(corpus, dis=False)
    return embedding, entity_index


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
    # 清洗数据集;前者适用于word2vec文本表征，后者适用于TF_IDF表征
    corpus1, corpus2 = preprocess_corpus_notDropEntity(corpus, involved_entity=involved_entity)
    embedding = G.distance_matrix(corpus1, dis=False)
    return corpus2, embedding, index2ind  # 返回数据集，返回word2vec的文本表征，返回index2ind


def build_topic_feature_sklearn(corpus, TopicNum=50):
    """
    mine the latent topics T
    :param corpus: 文本切分后的短语情况
    :param TopicNum: Topic数量
    :return:
    """
    alpha, beta = 0.1, 0.1
    lda = LatentDirichletAllocation(n_components=TopicNum, max_iter=100,
                                    learning_method='batch', n_jobs=-1,
                                    doc_topic_prior=alpha, topic_word_prior=beta
                                    )
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    lda_feature = lda.fit_transform(X)

    return pca.fit_transform(lda.components_), lda_feature


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


def remove(graph, features_text, features_index_BOWTFIDF, features_entity_TFIDF,
           features_entity_index_desc, topic_word, doc_topic, doc_idx_list,
           TopK_for_Topics=2):
    text_nodes, entity_nodes, topic_nodes = cnt_nodes(graph)
    feature = features_text
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
    idx_map_list = []
    for type_name in type_list:
        indexes = []
        if type_name == "text":
            for i in range(textShape[0]):
                ind = features_index[i]
                if (ind) not in text_nodes:
                    continue
                indexes.append(mapindex[ind])
        elif type_name == "entity":
            for i in range(entityShape[0]):
                name = features_entity_index[i]
                if name not in entity_nodes:
                    continue
                indexes.append(mapindex[name])
        else:
            for i in range(topic_num):
                topicName = topics[i]
                if topicName not in topic_nodes:
                    continue
                indexes.append(mapindex[topicName])
        # 索引匹配
        idx = np.stack(indexes)
        for i in idx:
            idx2type[type_name].add(i)
        idx_map = {j: i for i, j in enumerate(idx)}
        idx_map_list.append(idx_map)
    train, vali, test, alltext = sample("../Dataset/HGATN_train_data/", DATASETS="HGAT_data.csv", resample=False)
    # 将训练集、验证集、测试集的索引存储起来
    train = list(map(int, train))
    test = list(map(int, test))
    vali = list(map(int, vali))
    gnodes = set(graph.nodes())
    print(gnodes, mapindex)
    with open("../Dataset/HGATN_train_data/" + 'train.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in train if i in gnodes]))
    with open("../Dataset/HGATN_train_data/" + 'vali.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in vali if i in gnodes]))
    with open("../Dataset/HGATN_train_data/" + 'test.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in test if i in gnodes]))
    # 存储数据
    content = dict()
    for i in range(textShape[0]):
        ind = features_index[i]
        if (ind) not in text_nodes:
            continue
        content[ind] = feature[i, :].tolist()
    data = pd.read_csv("../Dataset/HGATN_train_data/" + 'HGAT_data.csv', lineterminator="\n")
    # 将事件类型加入
    for index, row in data.iterrows():
        if (row["tweet_id"]) not in text_nodes:
            continue
        content[row["tweet_id"]] += [row["event_id"]]
    # 存储数据
    with open("../Dataset/HGATN_train_data/" + 'HGAT.content.text', 'w') as f:
        for ind in content:
            f.write(str(mapindex[ind]) + '\t' + '\t'.join(map(str, content[ind])) + '\n')
    cache = len(content)
    print("共{}个文本".format(len(content)))
    # entity node
    content = dict()
    for i in range(entityShape[0]):
        name = features_entity_index[i]
        if name not in entity_nodes:
            continue
        content[name] = entityF[i, :].tolist() + ['entity']
    with open("../Dataset/HGATN_train_data/" + 'HGAT.content.entity', 'w') as f:
        for ind in content:
            f.write(str(mapindex[ind]) + '\t' + '\t'.join(map(str, content[ind])) + '\n')
    cache += len(content)
    print("共{}个实体".format(len(content)))
    # topic node
    content = dict()
    for i in range(topic_num):
        topicName = topics[i]
        if topicName not in topic_nodes:
            continue
        one_hot = [0] * topic_num
        one_hot[i] = 1
        content[topicName] = one_hot
        content[topicName] = topic_word[i].tolist() + ['topic']

    # 存储TOPIC信息
    with open("../Dataset/HGATN_train_data/" + 'HGAT.content.topic', 'w') as f:
        for ind in content:
            f.write(str(mapindex[ind]) + '\t' + '\t'.join(map(str, content[ind])) + '\n')
    cache += len(content)
    print("共{}个主题".format(len(content)))
    # exit()
    # adj matrix
    doneSet = set()
    edges_unordered = []
    for node in graphdict:
        for i in graphdict[node]:
            if (node, i) not in doneSet:
                edges_unordered.append([mapindex[node], mapindex[i]])
    # save adj matrix
    with open('../Dataset/HGATN_train_data/mapindex.txt', 'w') as f:
        for i in mapindex:
            f.write("{}\t{}\n".format(i, mapindex[i]))
    with open("../Dataset/HGATN_train_data/HGAT.cites", 'w') as f:
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
    exit()
    for i in range(len(mapindex)):
        edges_unordered.append([i, i])
    len_list = [len(idx2type[t]) for t in type_list]
    type2len = {t: len(idx2type[t]) for t in type_list}
    len_all = sum(len_list)
    adj_all = sp.lil_matrix(np.zeros((len_all, len_all)), dtype=np.float32)
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
    adj1 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:0]): sum(len_list[:1])]  # 文本和文本之间的关系
    adj2 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:1]): sum(len_list[:2])]  # 文本和话题之间的关系
    adj3 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:2]): sum(len_list[:3])]  # 文本和实体之间的关系
    adj4 = adj_all[sum(len_list[:2]): sum(len_list[:3]), sum(len_list[:2]): sum(len_list[:3])]  # 实体和实体之间的关系
    adj5 = np.matmul(adj3.todense(), adj4.todense())  # 更新后的文本和实体之间的关系
    print(adj1.shape, adj2.shape, adj5.shape)
    exit()
    # 从sparse变成tensor
    adj4 = np.matmul(adj1.todense(), adj3.todense())
    adj2 = adj2.todense()
    adj2 = np.matmul(adj2, adj2.transpose())
    adj4 = np.matmul(adj4, adj4.transpose())
    return [sp.csr_matrix(adj2), sp.csr_matrix(adj4)]


def build_information_network(res):
    """
    res 应该类似于一个json
    :param res:
    :return:
    """
    # 先将所有的数据 识别 实体
    print("step1")
    ans = get_sentence_entity(res)  # ans 中的第一列为id信息，第二列为实体信息
    print("step2")
    graph = build_graph(ans)  # 加载实体信息，构建图
    print("step3")
    # 为什么要用到这个函数？？？
    features_entity_desc, features_entity_index_desc = build_entity_feature_with_description(graph=graph)
    print("step4")
    # text_feature
    corpus_words, text_features, features_index = build_text_feature(ans, res)
    lda_model_word, doc_topic_distribution = build_topic_feature_sklearn(corpus=corpus_words)
    # 生成异构图
    ans = np.array(ans)
    print("text shape")
    print(text_features.shape, features_entity_desc.shape, lda_model_word.shape)
    np.savez("../Dataset/HGATN_train_data/feature.npz", text=text_features, entity=features_entity_desc, topic=lda_model_word)
    AL = remove(graph, text_features, features_index_BOWTFIDF=features_index,
                features_entity_TFIDF=features_entity_desc,
                features_entity_index_desc=features_entity_index_desc,
                topic_word=lda_model_word,
                doc_topic=doc_topic_distribution,
                doc_idx_list=ans[:, 0],
                TopK_for_Topics=2)
    # 获取文本信息
    return [text_features, lda_model_word, features_entity_desc], AL


def stream_information(res):
    ans = []
    labels_true = []
    cnt = 1
    for obj in res:
        if cnt > 1000:
            break
        cnt += 1
        info = {
                    "_id": int(obj["_id"]),
                    "text": obj["text"]
                }
        labels_true.append(obj['event_id'])
        ans.append(info)
    print("数据样本量：", cnt)
    feature, adj_list = build_information_network(ans)
    feature = torch.tensor(feature)
    feature = feature.to(torch.float32)
    # 构造图
    adj1 = dgl.from_scipy(adj_list[0])
    adj2 = dgl.from_scipy(adj_list[1])
    adj1 = dgl.add_self_loop(adj1)
    adj2 = dgl.add_self_loop(adj2)
    # 加载模型获得文本输入
    # output = HMODEL([adj1, adj2], feature)
    # embedding = HMODEL.write_emb.tolist()
    # 计算距离
    # X = pairwise_distances(embedding)
    # return X, labels_true


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
    # time_list = range_date("2012-10-10", "2012-11-07")
    # N = len(time_list)
    # # 调用数据查询方法
    # MG = MONGO("TwitterEvent2012", "tweets")
    # for i in range(N - 1):
    #     # 根据时间信息进行信息检索；返回的是游标，会随着遍历而移动
    #     response = MG.query(time_list[i], time_list[i + 1])
    #     print("start")
    #     stream_information(response)
    result = pd.read_csv('../Dataset/HGAT_train_data/HGAT_data.csv', lineterminator="\n")
    data = result[["tweet_id", "text\r"]]
    print(data.shape)
    # 按行遍历数据
    # 假设传入的也是类似于从数据库中读取的json格式
    ans = []
    for index, row in data.iterrows():
        info = {
            "_id": int(row["tweet_id"]),
            "text": row["text\r"],
        }
        ans.append(info)
    feature, adj_list = build_information_network(ans)