#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
import scipy.sparse as sp
import os


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
    print(idx_train)
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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if len(sparse_mx.nonzero()[0]) == 0:
        # 空矩阵
        r, c = sparse_mx.shape
        return torch.sparse.FloatTensor(r, c)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_tensor_to_sparse(dense_mx):
    return sparse_mx_to_torch_sparse_tensor(sp.coo.coo_matrix(dense_mx))


def load_data(path=None, dataset="HGAT"):
    # HIN包含三张类型的节点：文本、话题、实体
    type_list = ['text', 'topic', 'entity']
    type_have_label = 'text'
    features_list = []
    idx_map_list = []
    # 构造map， key为节点类型，内容为一个集合
    idx2type = {t: set() for t in type_list}

    for type_name in type_list:
        print('Loading {} content...'.format(type_name))
        print(path)
        print(type_name)
        indexes, features, labels = [], [], []
        # 读取数据
        with open("{}{}.content.{}".format(path, dataset, type_name)) as f:
            for line in tqdm(f):
                cache = line.strip().split('\t')
                indexes.append(np.array(cache[0], dtype=int))
                features.append(np.array(cache[1:-1], dtype=np.float32))
                labels.append(np.array([cache[-1]], dtype=str))
            # 连接数组
            features = np.stack(features)
            # 归一化
            features = normalize(features)
            features = torch.FloatTensor(np.array(features))
            features = dense_tensor_to_sparse(features)

            features_list.append(features)
        # 对标签类型进行处理
        if type_name == type_have_label:
            labels = np.stack(labels)
            # 独热码编码
            labels = encode_onehot(labels)
            Labels = torch.LongTensor(labels)
            print("label matrix shape: {}".format(Labels.shape))
        # 索引匹配
        idx = np.stack(indexes)
        for i in idx:
            idx2type[type_name].add(i)
        idx_map = {j: i for i, j in enumerate(idx)}
        idx_map_list.append(idx_map)
        print('done.')

    len_list = [len(idx2type[t]) for t in type_list]
    type2len = {t: len(idx2type[t]) for t in type_list}
    len_all = sum(len_list)
    # build graph
    print('Building graph...')
    adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]
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

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):
            adj_list[i1][i2] = sparse_mx_to_torch_sparse_tensor(
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                sum(len_list[:i2]): sum(len_list[:i2 + 1])]
            )

    print("Num of edges: {}".format(len(adj_all.nonzero()[0])))
    idx_train, idx_val, idx_test = load_divide_idx(path, idx_map_list[0])
    return adj_list, features_list, Labels, idx_train, idx_val, idx_test, idx_map_list[0]


def nll_loss(preds, y):
    """
    多分类损失函数
    :param preds:
    :param y:
    :return:
    """
    y = y.max(1)[1]
    return F.nll_loss(preds, y)


def makedirs(dirs: list):
    """
    生成文件夹
    :param dirs: 文件夹名称列表
    :return:
    """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

