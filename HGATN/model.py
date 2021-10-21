#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    """
    Semantic-level Attention
    将多种语义信息融合在一起
    """
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=True)  # 这里原先是False
        )

    def forward(self, z):
        """
        前向计算
        :param z: Node Level的语义嵌入
        :return:
        """
        w = self.project(z)  # tanh(Wz+b)
        beta = torch.softmax(w, dim=1)  # 经过softmax()得到语义权重
        return (beta * z).sum(1)   # 返回语义层的嵌入


class HANLayer(nn.Module):
    """
    实现Node Level和 Semantic-level 计算
    Arguments
    ---------
    num_meta_paths : 根据 metapath 产生的异构图数量
    in_size : 输入特征维度
    out_size : 输出特征维度
    layer_num_heads : 多头机制中，head数量
    dropout : Dropout机制中舍弃的概率
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout, low_dim, high_dim):
        super(HANLayer, self).__init__()
        self.num_meta_paths = num_meta_paths
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))  # 调用实现多头机制的GAT模型
        # 不同矩阵的维度转换
        self.trans = nn.Linear(low_dim, high_dim, bias=True)
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)

    def forward(self, gs, hs):
        """
        :param gs: list[DGLGraph]
        :param h: tensor, Input features
        :return: tensor, The output feature
        """
        semantic_embeddings = []
        for i, g in enumerate(gs):
            # 每次循环根据一个图返回一条meat-path上的嵌入
            semantic_embeddings.append(self.gat_layers[i](g, hs[i]).flatten(1))
        # semantic_embeddings 中包含的是不同路径的语义,应该进行一个语义的对齐
        # 假定话题是在最后一个语义上
        topic = semantic_embeddings.pop()
        # 转化维度
        semantic_embeddings.append(self.trans(topic))
        # 将不同路径上的语义进行聚合
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)   # (N, D * K)


class HGAN(nn.Module):
    """
    实现了一个HAN模型
     Arguments
    ---------
    num_meta_paths : 根据 metapath 产生的异构图数量
    in_size : 输入特征维度
    hidden_size : 隐藏层维度
    out_size : 输出特征维度
    num_heads : 每层layer中多头机制的数量
    dropout : Dropout机制中舍弃的概率
    """
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size,
                 num_heads, dropout, low_dim=200, high_dim=500):
        super(HGAN, self).__init__()
        self.write_emb = None
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout, low_dim, high_dim))
        # 这里使用一层MLP
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size, bias=True)

    def forward(self, g, h):
        # 多层HANLayer，不管更新学习的嵌入
        for gnn in self.layers:
            h = gnn(g, h)
        self.write_emb = h
        predict_result = self.predict(h)
        return F.log_softmax(predict_result, dim=1)

