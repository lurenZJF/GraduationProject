#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
func: 构建流式场景下的异构图
用HGAT表征文本时，无需保存处理后的数据，直接传入下一个函数；
"""
import sys
import dgl
import scipy.sparse as sp
from dgl.nn import GATConv
from sklearn.decomposition import PCA
import tqdm
sys.path.append('/home/dell/GraduationProject/')
from HGATN.utils import *
# pca模型统一到200维
pca = PCA(n_components=50)


def load_divide_idx(idx_map, path=None):
    # print(idx_map)
    # print(len(idx_map))
    # exit()
    # print(len(idx_map))
    # print(path)
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
    # print(idx_val)
    # print(idx_train)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test


def get_binary_mask(total_size, indices):
    # mask机制
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.bool()


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


def hom_load_data(path="../Dataset/Pre_data/", dataset="HGAT"):
    print('Loading data...', flush=True)
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
            for line in f:
                cache = line.strip().split('\t')
                indexes.append(np.array(cache[0], dtype=int))
                features.append(np.array(cache[1:-1], dtype=np.float32))
                labels.append(np.array([cache[-1]], dtype=str))
            # 连接数组
            features = np.stack(features)
            if type_name in ["entity", "text"]:
                features = pca.fit_transform(features)
            # 归一化
            features = normalize(features)
            features = torch.FloatTensor(np.array(features))
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
    adj1 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:0]): sum(len_list[:1])]  # 文本和文本之间的关系
    adj2 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:1]): sum(len_list[:2])]  # 文本和话题之间的关系
    adj3 = adj_all[sum(len_list[:0]): sum(len_list[:1]), sum(len_list[:2]): sum(len_list[:3])]  # 文本和实体之间的关系
    adj4 = adj_all[sum(len_list[:2]): sum(len_list[:3]), sum(len_list[:2]): sum(len_list[:3])]  # 实体和实体之间的关系
    adj5 = np.matmul(adj3.todense(), adj4.todense())  # 更新后的文本和实体之间的关系
    print(adj1.shape, adj2.shape, adj5.shape)
    for f in features_list:
        print(f.shape)
    # 构造最终的数据
    # 构造图
    adj1 = dgl.from_scipy(sp.csr_matrix(adj1))
    adj1 = dgl.add_self_loop(adj1)
    # topic和entity图中两者大小不一致
    edj2 = []
    for i in range(adj2.shape[0]):
        for j in range(adj2.shape[1]):
            edj2.append([j, i])
    edj5 = []
    for i in range(adj5.shape[0]):
        for j in range(adj5.shape[1]):
            edj5.append([j, i])
    adj2 = dgl.heterograph({("text", "+1", "topic"): edj2})
    adj5 = dgl.heterograph({("text", "+1", "topic"): edj5})
    num_nodes = adj1.number_of_nodes()  # 节点数量
    idx_train, idx_val, idx_test = load_divide_idx(idx_map_list[0])
    # mask操作
    train_mask = get_binary_mask(num_nodes, idx_train)
    test_mask = get_binary_mask(num_nodes, idx_test)
    val_mask = get_binary_mask(num_nodes, idx_val)
    return [adj1, adj2, adj5], features_list, Labels, train_mask, val_mask, test_mask


def h_load_data(path="../Dataset/HGATN_train_data/", dataset="HGAT"):
    print('Loading data...', flush=True)
    # HIN包含三张类型的节点：文本、话题、实体
    type_list = ['text', 'topic', 'entity']
    type_have_label = 'text'
    idx_map_list = []
    # 构造map， key为节点类型，内容为一个集合
    idx2type = {t: set() for t in type_list}
    for type_name in type_list:
        print('Loading {} content...'.format(type_name))
        print(path)
        print(type_name)
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
    # 记载图数据
    edges_unordered = np.genfromtxt("{}{}.cites".format("../Dataset/HGATN_train_data/", dataset), dtype=np.int32)
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
    # exit()
    # 读取特征数据
    all_feture = np.load("../Dataset/HGATN_train_data/feature.npz")
    features_list = [torch.FloatTensor(all_feture["text"]), torch.FloatTensor(all_feture["topic"]), torch.FloatTensor(all_feture["entity"])]
    for f in features_list:
        print(f.shape)
    # 构造最终的数据
    # 构造图
    adj1 = dgl.from_scipy(sp.csr_matrix(adj1))
    adj1 = dgl.add_self_loop(adj1)
    # topic和entity图中两者大小不一致
    edj2 = []
    for i in range(adj2.shape[0]):
        for j in range(adj2.shape[1]):
            edj2.append([j, i])
    edj5 = []
    for i in range(adj5.shape[0]):
        for j in range(adj5.shape[1]):
            edj5.append([j, i])
    adj2 = dgl.heterograph({("text", "+1", "topic"): edj2})
    adj5 = dgl.heterograph({("text", "+1", "topic"): edj5})
    num_nodes = adj1.number_of_nodes()  # 节点数量
    idx_train, idx_val, idx_test = load_divide_idx(idx_map_list[0], path="../Dataset/HGATN_train_data/")
    # mask操作
    train_mask = get_binary_mask(num_nodes, idx_train)
    test_mask = get_binary_mask(num_nodes, idx_test)
    val_mask = get_binary_mask(num_nodes, idx_val)
    return [adj1, adj2, adj5], features_list, Labels, train_mask, val_mask, test_mask


# if __name__ == "__main__":
#     g, features, label, train_mask, val_mask, test_mask = h_load_data()
# #     # gatconv = GATConv((4455, 200), 100, 50)
# #     # res = gatconv(g[1], (features[1], features[0]))
# #     # print(res.shape)








