#!/usr/bin/env python
# -*- coding:utf-8 -*-
import dgl
import scipy.sparse as sp
import torch
import sys
import numpy as np
sys.path.append('/home/dell/GraduationProject/')
# from Baseline.glove2vec import *
# from TextFiltering.twitter_preprocessor import *
# Cut = TwitterPreprocessor()
# G = GenerateWordVectors("../Static/glove2word2vec.txt")


# def build_feature():
#     # 将不同类型的节点统一到一个文本维度进行表征
#     result = pd.read_csv('../Dataset/HGAT_train_data/HGAT_data.csv', lineterminator="\n")
#     data = result[["tweet_id", "text\r"]]
#     # 按行遍历数据
#     # 假设传入的也是类似于从数据库中读取的json格式
#     token_w = []
#     for index, row in data.iterrows():
#         words = Cut.get_token(row["text\r"])
#         token_w.append(words)
#     # 文本feature
#     ans = G.distance_matrix(token_w, dis=False)
#     # return ans
#     ans = np.array(ans)
#     np.savez("feature.npz", feature=ans)


def encode_onehot(labels):
    """
    将标签转化为one-hot编码形式
    :param labels:
    :return:
    """
    classes = set(labels.T[0])
    # 矩阵[i,:]是仅保留第一维度的下标i的元素和第二维度所有元素，直白来看就是提取了矩阵的第i行
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels.T[0])),
                             dtype=np.int32)
    return labels_onehot


def load_divide_idx(path, idx_map):
    idx_train = []
    idx_val = []
    idx_test = []
    with open(path + 'train.map', 'r') as f:
        for line in f:
            idx_train.append(idx_map.get(int(line.strip('\n'))))
    with open(path + 'vali.map', 'r') as f:
        for line in f:
            idx_val.append(idx_map.get(int(line.strip('\n'))))
    with open(path + 'test.map', 'r') as f:
        for line in f:
            idx_test.append(idx_map.get(int(line.strip('\n'))))

    print("train, vali, test: ", len(idx_train), len(idx_val), len(idx_test))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test


def normalize(mx):
    """
    将特征归一化
    :param mx:
    :return:
    """
    # 对axis=1 这个维度上的数据求和
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_graph(path=None, dataset="HGAT"):
    # HIN包含三张类型的节点：文本、话题、实体
    type_list = ['text', 'topic', 'entity']
    type_have_label = 'text'
    idx_map_list = []
    # 构造map， key为节点类型，内容为一个集合
    idx2type = {t: set() for t in type_list}
    for type_name in type_list:
        indexes, labels = [], []
        # 读取数据
        with open("{}{}.content.{}".format(path, dataset, type_name)) as f:
            for line in f:
                cache = line.strip().split('\t')
                indexes.append(np.array(cache[0], dtype=int))
                labels.append(np.array([cache[-1]], dtype=str))
        # 对标签类型进行处理
        if type_name == type_have_label:
            labels = np.stack(labels)
            labels = encode_onehot(labels)  # 独热码编码
            Labels = torch.LongTensor(labels)
        # 索引匹配
        idx = np.stack(indexes)
        for i in idx:
            idx2type[type_name].add(i)
        idx_map = {j: i for i, j in enumerate(idx)}
        idx_map_list.append(idx_map)
    #
    len_list = [len(idx2type[t]) for t in type_list]
    type2len = {t: len(idx2type[t]) for t in type_list}
    len_all = sum(len_list)
    # build graph
    print('Building graph...')
    # 记载图数据
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
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
    adj1 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:1]): sum(len_list[:2])]
    adj2 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:2]): sum(len_list[:3])]
    adj3 = adj_all[sum(len_list[:1]): sum(len_list[:2]), sum(len_list[:1]): sum(len_list[:2])]
    # 从sparse变成tensor
    adj4 = np.matmul(adj1.todense(), adj3.todense())
    adj2 = adj2.todense()
    adj2 = np.matmul(adj2, adj2.transpose())
    adj4 = np.matmul(adj4, adj4.transpose())

    idx_train, idx_val, idx_test = load_divide_idx(path, idx_map_list[0])
    return [sp.csr_matrix(adj2), sp.csr_matrix(adj4)], Labels, idx_train, idx_val, idx_test, idx_map_list[0]


def get_binary_mask(total_size, indices):
    # mask机制
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.bool()


def load_data():
    print('Loading data...', flush=True)
    # 加载特征数据
    data = np.load("feature.npz")
    features = data["feature"]
    # 加载图数据等
    graph, label, idx_train, idx_val, idx_test, idx_map_list = load_graph("../HGAT/outdata/")
    # 转化为tensor形式
    features = torch.FloatTensor(features)
    print(graph[0].shape)
    # 构造图
    adj1 = dgl.from_scipy(graph[0])
    adj2 = dgl.from_scipy(graph[1])
    adj1 = dgl.add_self_loop(adj1)
    adj2 = dgl.add_self_loop(adj2)
    num_nodes = adj1.number_of_nodes()  # 节点数量
    # mask操作
    train_mask = get_binary_mask(num_nodes, idx_train)
    test_mask = get_binary_mask(num_nodes, idx_test)
    val_mask = get_binary_mask(num_nodes, idx_val)
    return [adj1, adj2], features, label, train_mask, val_mask, test_mask















