#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
func: 构建HGAT模型
"""
import sys
sys.path.append('/home/penghao/Project/')
from HGAT.layers import *
from torch.nn.parameter import Parameter
from functools import reduce


class HGAT(nn.Module):
    """
    HGAT模型

    """
    def __init__(self, nfeat_list, nhid, nclass, dropout,
                 type_attention=True, node_attention=True,
                 gamma=0.1, sigmoid=False, orphan=True,
                 write_emb=True
                 ):
        super(HGAT, self).__init__()
        self.sigmoid = sigmoid
        self.type_attention = type_attention
        self.node_attention = node_attention

        self.write_emb = write_emb
        if self.write_emb:
            self.emb = None
            self.emb2 = None
        self.nonlinear = F.relu_
        self.nclass = nclass  # 有监督分类：类别数量
        self.ntype = len(nfeat_list)  # 节点类别数量

        dim_1st = nhid  # 隐藏层节点数据
        dim_2nd = nclass  # 输出类别数量
        if orphan:
            dim_2nd += self.ntype - 1

        self.gc2 = nn.ModuleList()
        if not self.node_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append(GraphConvolution(nfeat_list[t], dim_1st, bias=False))
                self.bias1 = Parameter(torch.FloatTensor(dim_1st))
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(nfeat_list, dim_1st, gamma=gamma)
        self.gc2.append(GraphConvolution(dim_1st, dim_2nd, bias=True))
        # 当self.node_attention = True时
        # self.gc1 = GraphAttentionConvolution(nfeat_list, dim_1st, gamma=gamma)
        # self.gc2 = [GraphAttentionConvolution(dim_1st, dim_2nd, bias=True)]
        # 两层注意力机制
        if self.type_attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append(SelfAttention(dim_1st, t, 50))
                self.at2.append(SelfAttention(dim_2nd, t, 50))
        # self.at1 有三个attention 机制，是不同类型的； self.at2同理
        self.dropout = dropout

    def forward(self, x_list, adj_list, adj_all=None):
        x0 = x_list
        # print("输入的特征长度", len(x0))
        if not self.node_attention:
            x1 = [None for _ in range(self.ntype)]
            # First Layer
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    idx = t2
                    x_t1.append(self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1)
                if self.type_attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))
                else:
                    x_t1 = reduce(torch.add, x_t1)

                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)
            # print("gc1之后的输出维度：", len(x1_in))
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]
                if self.type_attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]

        x2 = [None for _ in range(self.ntype)]
        # Second Layer
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                idx = 0
                x_t1.append(self.gc2[idx](x1[t2], adj_list[t1][t2]))
            if self.type_attention:
                x_t1, weights = self.at2[t1](torch.stack(x_t1, dim=1))
            else:
                x_t1 = reduce(torch.add, x_t1)

            x2[t1] = x_t1
            if self.write_emb and t1 == 0:
                self.emb2 = x2[t1]

            # output layer
            if self.sigmoid:
                x2[t1] = torch.sigmoid(x_t1)
            else:
                x2[t1] = F.log_softmax(x_t1, dim=1)
        return x2